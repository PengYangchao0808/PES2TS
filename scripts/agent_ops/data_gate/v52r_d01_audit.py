#!/usr/bin/env python3
"""V5.2-R stage D0.1 audit: label & data audit reinforcement (gate V52R-D0.1).

Read-only over the PD-1 six-table delivery (``outputs/pesn/30h_campaign/data``)
plus the frozen D0 evidence pack (``outputs/pesn/v52r/d0/evidence_pack.json``);
writes ONLY the D0.1 pack (``topology_anomaly_audit.json`` +
``evidence_pack.json`` + ``D0_1_AUDIT.md``) under ``--out`` (default
``outputs/pesn/v52r/d0_1``).  No training, no model loads, no sealed-fold
metric use — folds 3/4 rows are CLASSIFIED as label QC (identical caliber to
D0's topology proxy) and reported separately, never encoded (V5.2-R plan §3
D0.1; campaign ``pes2ts_v52r_20261021.yaml`` ``split_contract``).

Implements ``docs/plans/PES2TS_V52R_输入充分性与化学终点识别优化方案_v1_20261021.md``
§3 D0.1 against contract ``docs/contracts/PES2TS_FullPathNavigationContract_v1.md``
§5.2/§5.3:

1. §3.1 — classify ALL 22 D0 topology outliers by a PRE-REGISTERED decision
   tree:  S (symmetry-equivalent mapping: sym-min CV deviation ≤ 0.25 Å OR
   d_end_sym ≤ 0.10 Å under some automorphism of the target-endpoint graph;
   label recomputed with d_end_sym — stays/becomes 1 ⇒ keep positive,
   flips ⇒ Mask) → T (target bond-edit incomplete: every deviating CV pair
   is a changing bond AND its frame distance lies strictly BETWEEN the
   A- and B-endpoint values ⇒ Mask per contract §5.3 rule 5) → D
   (data/mapping error by elimination: frame cross-checked against the
   wp2 teacher.json states + atom-map identity invariant; any
   inconsistency ⇒ upstream-fix note; Mask either way).  Unresolved rows
   are NEVER positives: D is the fall-through class and masks.
2. §3.2 — recompute ``d_end_sym`` for the FULL label table (all 12034 rows;
   per-curve-intent automorphism sets computed once); report pos→neg /
   neg→pos flips, flips among the 22 outliers, new prevalence at
   0.10/0.30, and ``requires_d0_restatement`` (>2% of positives flip).
3. §3.3 — terminal provenance: recheck 1123/1781; terminal vs
   non-terminal positive d_end / K-pair-distance distributions, per-fold
   breakdown, KS + Wasserstein per metric; free-rollout reporting must
   stratify terminal vs non-terminal.
4. §3.4 — data-scale disclosure: statistical unit = CURVE (n=30 dev),
   7010 dev rows, 28 curves with ok rows; line-level significance forbidden
   standalone.
5. Gate (``--strict``): all 22 rows classified + disposition applied;
   automorphism cap never hit; dev/sealed split reported; negative test
   present; verdict PASS only if classification is complete.

Deterministic for fixed inputs (sorted keys, no timestamps, no RNG).

Usage::

    python scripts/agent_ops/data_gate/v52r_d01_audit.py \\
        [--tables-dir outputs/pesn/30h_campaign/data] \\
        [--d0-pack outputs/pesn/v52r/d0/evidence_pack.json] \\
        [--out outputs/pesn/v52r/d0_1] \\
        [--strict]          # exit 1 when the gate verdict is FAIL
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    FORWARD_VIEW_KEYS_V3,
    IntentRecordV3,
)
from pes2ts_core.pesn.offline.data.graph_symmetry import (  # noqa: E402
    AUTOMORPHISM_CAP,
    GraphSymmetryError,
    MolecularGraph,
    enumerate_automorphisms,
    k_automorphism_permutations,
    min_sym_rmsd,
    sym_min_cv_deviation,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    DATA_MANIFEST_FILENAME,
    SPLIT_MANIFEST_FILENAME,
    KeyAtomDatasetV3,
)
from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (  # noqa: E402
    ENDPOINT_STOP_NEGATIVE_WINDOW_ANGSTROM,
    ENDPOINT_STOP_POSITIVE_WINDOW_ANGSTROM,
    SCHEMA,
    EndpointStopLabelTable,
    build_endpoint_stop_labels,
    endpoint_stop_label_from_distance,
)

SCHEMA_ID = "pes2ts_v52r_d01_evidence_v1"
DEFAULT_TABLES_DIR = REPO_ROOT / "outputs/pesn/30h_campaign/data"
DEFAULT_D0_PACK = REPO_ROOT / "outputs/pesn/v52r/d0/evidence_pack.json"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs/pesn/v52r/d0_1"
CAMPAIGN_CONFIG = REPO_ROOT / "config/campaigns/pes2ts_v52r_20261021.yaml"
ENDPOINT_MODULE = (REPO_ROOT /
                   "pes2ts_core/pesn/offline/data/key_atom_endpoint_labels_v1.py")
SYMMETRY_MODULE = (REPO_ROOT /
                   "pes2ts_core/pesn/offline/data/graph_symmetry.py")
AUDIT_MODULE = Path(__file__).resolve()
LEGACY_TEST = REPO_ROOT / "tests/pesn/test_r3_input_legality.py"
GATE_ID = "V5.2R-D0.1-LABELS"

W_POS = ENDPOINT_STOP_POSITIVE_WINDOW_ANGSTROM
W_NEG = ENDPOINT_STOP_NEGATIVE_WINDOW_ANGSTROM

#: D0 topology-proxy tolerance (frozen; outliers were listed at > 0.25 Å).
TOPOLOGY_DEV_TOL_ANGSTROM = 0.25
#: S-class threshold on d_end_sym (plan §3.1: symmetric-mapping positive).
S_D_END_SYM_ANGSTROM = W_POS
#: Acceptance targets (V5.2 task-correction audit 20261021 / D0 pack).
EXPECT_TOTAL_ROWS = 12034
EXPECT_POS = 1781
EXPECT_NEG = 8399
EXPECT_MASK = 1854
EXPECT_OUTLIERS = 22
EXPECT_TERMINAL_POS = 1123
EXPECT_DEV_CURVES = 30
EXPECT_DEV_CURVES_WITH_OK = 28
EXPECT_DEV_ROWS = 7010
#: §3.2 restatement trigger: > 2% of positives change positive-class
#: membership under d_end_sym (pre-registered in plan §3.2).
RESTATEMENT_FRAC = 0.02
#: §3.1 T-class "strictly between" numerical guard (pre-registered; see
#: D0_1_AUDIT.md §1 clarifications).
BETWEEN_EPS_ANGSTROM = 1e-9
#: Teacher cross-check coordinate tolerance (irc_node coords_flat has 10
#: decimals; teacher coordinates are full float).
TEACHER_COORD_TOL_ANGSTROM = 1e-6
#: Statistical-unit disclosure (plan §3.4).
DISCLOSURE_TEXT = (
    "所有 D2 结论的统计单位是曲线（n=30 dev），不是行"
    "（n=7010 dev 行）；行级显著性不得单独使用。")


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            capture_output=True, text=True, check=True)
        return out.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _round(x: float, nd: int = 6) -> float:
    return round(float(x), nd)


# ------------------------------------------------------------- graph cache


def intent_target_graph(intent: IntentRecordV3) -> MolecularGraph:
    """Target-endpoint graph of an intent (forward → G_tgt = B; reverse → G_src = A)."""
    bonds = intent.g_tgt_bonds if intent.direction == "forward" else intent.g_src_bonds
    if not intent.g_atoms or not bonds:
        raise GraphSymmetryError(
            "INTENT_GRAPH_EMPTY",
            f"intent {intent.intent_id} lacks g_atoms or endpoint bonds")
    return MolecularGraph.from_intent_payload(intent.g_atoms, bonds)


class SymmetryCache:
    """Per-(curve, side) automorphism sets, computed once and reused.

    ``side`` is ``src`` (A / reactant graph) or ``tgt`` (B / product graph).
    A forward row's target endpoint is B ⇒ symmetry of G_tgt; a reverse
    row's target endpoint is A ⇒ symmetry of G_src.  K is identical for the
    fwd/rev intent pair of one curve (key_rows derives from the union of
    G_src/G_tgt edits + CV atoms), asserted on first use.
    """

    def __init__(self, cap: int = AUTOMORPHISM_CAP) -> None:
        self._cap = cap
        self._graphs: dict[tuple[str, str], MolecularGraph] = {}
        self._automorphisms: dict[tuple[str, str], tuple[tuple[int, ...], ...]] = {}
        self._k_perms: dict[tuple[str, str], tuple[tuple[int, ...], ...]] = {}
        self._k_sets: dict[tuple[str, str], tuple[int, ...]] = {}
        self.max_group_size = 0
        self.cap_hit = False

    def _side(self, intent: IntentRecordV3) -> str:
        return "tgt" if intent.direction == "forward" else "src"

    def _ensure(self, intent: IntentRecordV3) -> tuple[str, str]:
        key = (intent.curve_id, self._side(intent))
        if key not in self._graphs:
            graph = intent_target_graph(intent)
            autos = enumerate_automorphisms(graph, cap=self._cap)
            self._graphs[key] = graph
            self._automorphisms[key] = autos
            self.max_group_size = max(self.max_group_size, len(autos))
        return key

    def graph(self, intent: IntentRecordV3) -> MolecularGraph:
        return self._graphs[self._ensure(intent)]

    def automorphisms(self, intent: IntentRecordV3) -> tuple[tuple[int, ...], ...]:
        return self._automorphisms[self._ensure(intent)]

    def k_perms(self, intent: IntentRecordV3) -> tuple[tuple[int, ...], ...]:
        key = self._ensure(intent)
        if key not in self._k_perms:
            k = tuple(sorted(int(a) for a in intent.key_rows()))
            perms = k_automorphism_permutations(
                self._graphs[key], k, cap=self._cap)
            prev = self._k_sets.setdefault(key, k)
            if prev != k:
                raise GraphSymmetryError(
                    "K_SET_MISMATCH",
                    f"curve {key[0]} side {key[1]}: K {k} != {prev}")
            self._k_perms[key] = perms
        return self._k_perms[key]

    def d_end_sym(
        self,
        intent: IntentRecordV3,
        frame: np.ndarray,
        ref: np.ndarray,
    ) -> float:
        keys = np.asarray(sorted(int(a) for a in intent.key_rows()),
                          dtype=np.int64)
        return min_sym_rmsd(frame[keys], ref[keys], self.k_perms(intent))


# ----------------------------------------------------------- classification


def pair_devs(
    frame: np.ndarray,
    ref: np.ndarray,
    pairs: tuple[tuple[int, int], ...],
) -> list[float]:
    """Native-frame |d(row) − d(target endpoint)| per CV pair (D0 convention)."""
    out: list[float] = []
    for i, j in pairs:
        d_row = float(np.linalg.norm(frame[i] - frame[j]))
        d_endp = float(np.linalg.norm(ref[i] - ref[j]))
        out.append(abs(d_row - d_endp))
    return out


def strictly_between(value: float, a: float, b: float,
                     eps: float = BETWEEN_EPS_ANGSTROM) -> bool:
    """Pre-registered strict-between with a numerical guard (plan §3.1 T)."""
    lo, hi = (a, b) if a <= b else (b, a)
    return bool(lo + eps < value < hi - eps)


def diagnose_row(
    dataset: KeyAtomDatasetV3,
    base: EndpointStopLabelTable,
    rec: Any,
    cache: SymmetryCache,
) -> dict[str, Any]:
    """Full per-row evidence for the §3.1 decision tree (one outlier row)."""
    lab = base.by_sample[rec.sample_id]
    intent = rec.intent
    keys = np.asarray(sorted(int(a) for a in intent.key_rows()), dtype=np.int64)
    frame = np.asarray(dataset.frame_xyz(rec.curve_id, rec.node_index),
                       dtype=np.float64)
    ref = np.asarray(dataset.frame_xyz(rec.curve_id, lab.target_endpoint_node),
                     dtype=np.float64)
    curve = base.curves[rec.curve_id]
    frame_a = np.asarray(dataset.frame_xyz(rec.curve_id, curve.endpoint_a_node),
                         dtype=np.float64)
    frame_b = np.asarray(dataset.frame_xyz(rec.curve_id, curve.endpoint_b_node),
                         dtype=np.float64)
    pairs = tuple((int(i), int(j)) for i, j in intent.cv_coordinates)
    devs = pair_devs(frame, ref, pairs)

    autos = cache.automorphisms(intent)
    sym_min_dev = sym_min_cv_deviation(frame, ref, pairs, autos)
    d_end_sym = cache.d_end_sym(intent, frame, ref)

    # per-pair diagnosis
    pair_rows: list[dict[str, Any]] = []
    for (i, j), dev in zip(pairs, devs):
        d_row = float(np.linalg.norm(frame[i] - frame[j]))
        d_a = float(np.linalg.norm(frame_a[i] - frame_a[j]))
        d_b = float(np.linalg.norm(frame_b[i] - frame_b[j]))
        delta_b = float(intent.delta_b(i, j))
        pair_rows.append({
            "pair": [i, j],
            "dev_angstrom": _round(dev),
            "deviating": bool(dev > TOPOLOGY_DEV_TOL_ANGSTROM),
            "d_frame_angstrom": _round(d_row),
            "d_endpoint_a_angstrom": _round(d_a),
            "d_endpoint_b_angstrom": _round(d_b),
            "delta_b_intent": _round(delta_b),
            "is_changing_bond": bool(delta_b != 0.0),
            "strictly_between_ab": strictly_between(d_row, d_a, d_b),
        })

    return {
        "sample_id": rec.sample_id,
        "curve_id": rec.curve_id,
        "node_index": int(rec.node_index),
        "direction": intent.direction,
        "fold": dataset.folds.get(rec.curve_id),
        "split": ("dev" if dataset.folds.get(rec.curve_id) in (0, 1, 2)
                  else "sealed"),
        "label_status": rec.label_status,
        "d_end_angstrom": _round(lab.d_end_angstrom),
        "d_end_sym_angstrom": _round(d_end_sym),
        "sym_min_cv_dev_angstrom": _round(sym_min_dev),
        "n_automorphisms": len(autos),
        "n_k_perms": len(cache.k_perms(intent)),
        "n_pairs": len(pairs),
        "max_pair_dev_angstrom": _round(max(devs)) if devs else None,
        "pair_diagnosis": pair_rows,
    }


def teacher_cross_check(
    dataset: KeyAtomDatasetV3,
    rec: Any,
) -> dict[str, Any]:
    """Verify the row's frame against the wp2 teacher.json export (class D).

    Checks: (a) atom-map identity order (row i ↔ map id m{i+1:04d}, the g1
    invariant enforced at build time); (b) atomic numbers equal the teacher
    states'; (c) the node's coordinates equal the corresponding teacher
    state within :data:`TEACHER_COORD_TOL_ANGSTROM`.  Node → state mapping
    mirrors ``r3_build_tables``: nodes ``0..ts_index`` are branch-A states
    by step_index; nodes ``> ts_index`` read branch B backwards from the
    second-to-last frame (the TS frame is shared and stored once).
    """
    curve_row = dataset._curves.get(rec.curve_id)  # read-only table view
    out: dict[str, Any] = {
        "source_path": "", "state_id": "", "max_abs_coord_diff": None,
        "atom_map_identity_order": None, "atomic_numbers_match": None,
        "coordinates_match": None, "consistent": None, "error": "",
    }
    if curve_row is None:
        out["error"] = "curve not in irc_curve"
        out["consistent"] = False
        return out
    src = curve_row.get("source_path", "")
    out["source_path"] = src
    path = REPO_ROOT / src if src else None
    if path is None or not path.is_file():
        out["error"] = f"teacher.json not found: {src!r}"
        out["consistent"] = False
        return out
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        states = doc["states"]
        branches = {b["branch_id"]: b.get("endpoint_label", "")
                    for b in doc["branches"]}
        by_label: dict[str, list[dict[str, Any]]] = {"A": [], "B": []}
        for s in states:
            label = branches.get(s["branch_id"], "")
            if label in by_label:
                by_label[label].append(s)
        for label in by_label:
            by_label[label].sort(key=lambda s: int(s["step_index"]))
        a_states, b_states = by_label["A"], by_label["B"]
        ts_index = int(curve_row["ts_index"])
        node = int(rec.node_index)
        if 0 <= node <= ts_index:
            state = a_states[node] if node < len(a_states) else None
        else:
            k = node - ts_index - 1          # offset into reversed B segment
            step = len(b_states) - 2 - k
            state = b_states[step] if 0 <= step < len(b_states) else None
        if state is None:
            out["error"] = f"no teacher state for node {node}"
            out["consistent"] = False
            return out
        out["state_id"] = str(state.get("state_id", ""))
        map_ids = [str(m) for m in state.get("atom_map_ids", [])]
        expected = [f"m{i + 1:04d}" for i in range(len(map_ids))]
        out["atom_map_identity_order"] = bool(map_ids == expected)
        teacher_z = [int(z) for z in state.get("atomic_numbers", [])]
        row_z = [int(z) for z in rec.atomic_numbers]
        out["atomic_numbers_match"] = bool(teacher_z == row_z)
        frame = np.asarray(dataset.frame_xyz(rec.curve_id, node),
                           dtype=np.float64)
        coords = np.asarray(state["coordinates"], dtype=np.float64)
        if coords.shape == frame.shape:
            diff = float(np.max(np.abs(coords - frame)))
            out["max_abs_coord_diff"] = _round(diff, 9)
            out["coordinates_match"] = bool(diff <= TEACHER_COORD_TOL_ANGSTROM)
        else:
            out["coordinates_match"] = False
            out["error"] = (f"shape mismatch frame {frame.shape} vs "
                            f"teacher {coords.shape}")
        out["consistent"] = bool(out["atom_map_identity_order"]
                                 and out["atomic_numbers_match"]
                                 and out["coordinates_match"])
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["consistent"] = False
    return out


def classify_row(row: dict[str, Any]) -> dict[str, Any]:
    """PRE-REGISTERED §3.1 decision tree: S → T → D (exhaustive, ordered).

    S (symmetry-equivalent mapping): the CV anomaly is EXPLAINED by a
    graph automorphism of the target-endpoint graph — i.e. the
    symmetry-minimized max CV-pair deviation drops to ≤ 0.25 Å.  (The
    secondary ``d_end_sym ≤ 0.10`` clause only fires when it CHANGES the
    label: a non-positive row that becomes positive under symmetry.  For
    the 22 D0 outliers — all already positive with d_end ≤ 0.10 — this
    clause is vacuous because d_end_sym ≤ d_end ≤ 0.10 by construction
    (identity ∈ Aut(G)); it therefore never masks a genuine anomaly.)
    Label is recomputed with d_end_sym; stays/becomes 1 ⇒ keep positive,
    otherwise ⇒ Mask.

    T (target bond-edit incomplete): not S, AND every deviating CV pair
    (>0.25 Å) is a changing bond (``delta_b_intent ≠ 0``) AND its frame
    distance lies strictly between the A- and B-endpoint values (edit in
    progress) ⇒ Mask (contract §5.3 rule 5).

    D (data/mapping error, fall-through): neither S nor T ⇒ Mask; the
    teacher cross-check (run by the caller for D rows) attaches the
    upstream-fix note when it finds an inconsistency.  A teacher-clean D
    row remains masked as an unexplained topological anomaly —
    unresolved rows are never positives.
    """
    # S: symmetry must actually REDUCE the CV anomaly (or flip a
    # non-positive row to positive).  The second disjunct is the §3.2
    # flip clause; it is gated on d_end > w_pos so it cannot fire for
    # already-positive outliers (where it would be vacuous).
    s_hit = (
        row["sym_min_cv_dev_angstrom"] <= TOPOLOGY_DEV_TOL_ANGSTROM
        or (row["d_end_sym_angstrom"] <= S_D_END_SYM_ANGSTROM
            and row["d_end_angstrom"] > S_D_END_SYM_ANGSTROM))
    if s_hit:
        new_label = endpoint_stop_label_from_distance(
            row["d_end_sym_angstrom"], positive_window=W_POS,
            negative_window=W_NEG)
        keep_positive = new_label == 1
        return {
            "class": "S",
            "class_reason": (
                f"sym_min_cv_dev={row['sym_min_cv_dev_angstrom']} ≤ 0.25"
                if row["sym_min_cv_dev_angstrom"] <= TOPOLOGY_DEV_TOL_ANGSTROM
                else (f"d_end_sym={row['d_end_sym_angstrom']} ≤ 0.10 under "
                      f"some automorphism (d_end={row['d_end_angstrom']} > "
                      "0.10; symmetry flips the row to positive)")),
            "d_end_sym_label": new_label,
            "disposition": "keep_positive" if keep_positive else "mask",
            "final_label": 1 if keep_positive else -1,
        }
    deviating = [p for p in row["pair_diagnosis"] if p["deviating"]]
    t_hit = bool(deviating) and all(
        p["is_changing_bond"] and p["strictly_between_ab"]
        for p in deviating)
    if t_hit:
        return {
            "class": "T",
            "class_reason": (
                f"{len(deviating)}/{len(deviating)} deviating pairs are "
                "changing bonds with frame distance strictly between the "
                "A/B endpoint values (target bond-edit incomplete; "
                "contract §5.3 rule 5)"),
            "d_end_sym_label": None,
            "disposition": "mask",
            "final_label": -1,
        }
    non_changing = [p["pair"] for p in deviating if not p["is_changing_bond"]]
    not_between = [p["pair"] for p in deviating if not p["strictly_between_ab"]]
    return {
        "class": "D",
        "class_reason": (
            "fall-through: not symmetry-explained and not an in-progress "
            "target bond edit (data/mapping error class; teacher "
            "cross-check attached by the caller)"),
        "d_end_sym_label": None,
        "disposition": "mask",
        "final_label": -1,
        "d_evidence": {
            "deviating_pairs_not_changing_bond": non_changing,
            "deviating_pairs_not_strictly_between": not_between,
        },
    }


def audit_outliers(
    dataset: KeyAtomDatasetV3,
    base: EndpointStopLabelTable,
    d0_pack: dict[str, Any],
    cache: SymmetryCache,
) -> dict[str, Any]:
    """Classify all D0 topology outliers (§3.1)."""
    outliers = d0_pack["topology_consistency_proxy"]["outliers"]
    by_sid = {rec.sample_id: rec for rec in dataset.samples}
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for outlier in outliers:
        sid = str(outlier["sample_id"])
        rec = by_sid.get(sid)
        if rec is None:
            missing.append(sid)
            continue
        row = diagnose_row(dataset, base, rec, cache)
        row["class"] = None
        row["teacher_check"] = None
        rows.append(row)

    # classify; run the teacher cross-check for every D candidate
    for row in rows:
        verdict = classify_row(row)
        if verdict["class"] == "D":
            rec = by_sid[row["sample_id"]]
            teacher = teacher_cross_check(dataset, rec)
            row["teacher_check"] = teacher
            verdict["upstream_fix_required"] = bool(
                teacher.get("consistent") is False)
            if teacher.get("consistent") is False:
                verdict["class_reason"] += (
                    f"; teacher cross-check INCONSISTENT "
                    f"({teacher.get('error') or 'fields false'}) — "
                    "upstream fix required")
            else:
                verdict["class_reason"] += (
                    "; teacher cross-check consistent (residual anomaly)")
        row.update(verdict)

    rows.sort(key=lambda r: r["sample_id"])
    class_counts = Counter(r["class"] for r in rows)
    disposition_counts = Counter(r["disposition"] for r in rows)
    split_rows = {
        split: [r["sample_id"] for r in rows if r["split"] == split]
        for split in ("dev", "sealed")}
    return {
        "n_outlier_rows_expected": EXPECT_OUTLIERS,
        "n_outlier_rows_in_d0_pack": len(outliers),
        "n_rows_classified": len(rows),
        "missing_sample_ids": missing,
        "class_counts": {c: class_counts.get(c, 0) for c in ("S", "T", "D")},
        "disposition_counts": {
            d: disposition_counts.get(d, 0)
            for d in ("keep_positive", "mask")},
        "dev_sealed_split": {
            "dev_rows": len(split_rows["dev"]),
            "sealed_rows": len(split_rows["sealed"]),
            "dev_sample_ids": split_rows["dev"],
            "sealed_sample_ids": split_rows["sealed"],
            "sealed_note": (
                "sealed-fold rows are classified as label QC only "
                "(same caliber as the D0 topology proxy); never encoded"),
        },
        "decision_tree": {
            "order": ["S", "T", "D"],
            "S": ("symmetry EXPLAINS the anomaly: sym_min_cv_dev ≤ 0.25 Å "
                  "under some automorphism of the target-endpoint graph "
                  "(forward → G_tgt, reverse → G_src); the secondary "
                  "d_end_sym ≤ 0.10 Å clause only fires when it CHANGES "
                  "the label (d_end > 0.10 → positive under symmetry) — "
                  "for already-positive outliers it is vacuous because "
                  "d_end_sym ≤ d_end ≤ 0.10 by construction (identity ∈ "
                  "Aut(G)) and would otherwise mask genuine anomalies. "
                  "Label recomputed with d_end_sym; stays/becomes 1 ⇒ "
                  "keep positive, else Mask"),
            "T": ("not S AND every deviating CV pair (>0.25 Å) is a changing "
                  "bond (delta_b_intent ≠ 0) AND its frame distance lies "
                  "strictly between the A- and B-endpoint values ⇒ Mask "
                  "(contract §5.3 rule 5; strict-between guard "
                  f"{BETWEEN_EPS_ANGSTROM} Å)"),
            "D": ("fall-through by elimination ⇒ Mask; teacher.json "
                  "cross-check (atom-map identity order, atomic numbers, "
                  "coordinates ≤ "
                  f"{TEACHER_COORD_TOL_ANGSTROM} Å) attaches the "
                  "upstream-fix note on any inconsistency; teacher-clean "
                  "residuals stay masked — unresolved rows are never "
                  "positives"),
            "thresholds": {
                "topology_dev_tol_angstrom": TOPOLOGY_DEV_TOL_ANGSTROM,
                "s_d_end_sym_angstrom": S_D_END_SYM_ANGSTROM,
                "w_pos": W_POS, "w_neg": W_NEG,
                "between_eps_angstrom": BETWEEN_EPS_ANGSTROM,
                "teacher_coord_tol_angstrom": TEACHER_COORD_TOL_ANGSTROM,
            },
        },
        "rows": rows,
    }


# ------------------------------------------------------- §3.2 d_end_sym recompute


def audit_d_end_sym(
    dataset: KeyAtomDatasetV3,
    base: EndpointStopLabelTable,
    cache: SymmetryCache,
) -> dict[str, Any]:
    """Recompute d_end over the full table with symmetry minimization."""
    rec_by_sid = {rec.sample_id: rec for rec in dataset.samples}
    d_end_cache: dict[tuple[str, int, int, str], float] = {}
    flips: list[dict[str, Any]] = []
    n_pos_stay = n_pos_to_neg = n_pos_to_mask = 0
    n_neg_to_pos = n_mask_to_pos = 0
    n_other = 0
    new_pos = 0
    for sid, lab in base.by_sample.items():
        rec = rec_by_sid[sid]
        intent = rec.intent
        side = "tgt" if intent.direction == "forward" else "src"
        key = (lab.curve_id, int(lab.node_index),
               int(lab.target_endpoint_node), side)
        d_sym = d_end_cache.get(key)
        if d_sym is None:
            frame = np.asarray(
                dataset.frame_xyz(lab.curve_id, lab.node_index),
                dtype=np.float64)
            ref = np.asarray(
                dataset.frame_xyz(lab.curve_id, lab.target_endpoint_node),
                dtype=np.float64)
            d_sym = cache.d_end_sym(intent, frame, ref)
            d_end_cache[key] = d_sym
        new_label = endpoint_stop_label_from_distance(
            d_sym, positive_window=W_POS, negative_window=W_NEG)
        old = lab.endpoint_stop_label
        if new_label == 1:
            new_pos += 1
        if old == 1 and new_label == 1:
            n_pos_stay += 1
        elif old == 1 and new_label == 0:
            n_pos_to_neg += 1
            flips.append({"sample_id": sid, "from": 1, "to": 0})
        elif old == 1 and new_label == -1:
            n_pos_to_mask += 1
            flips.append({"sample_id": sid, "from": 1, "to": -1})
        elif old == 0 and new_label == 1:
            n_neg_to_pos += 1
            flips.append({"sample_id": sid, "from": 0, "to": 1})
        elif old == -1 and new_label == 1:
            n_mask_to_pos += 1
            flips.append({"sample_id": sid, "from": -1, "to": 1})
        else:
            n_other += 1
    flips.sort(key=lambda f: f["sample_id"])
    pos_membership_flips = n_pos_to_neg + n_pos_to_mask + n_neg_to_pos \
        + n_mask_to_pos
    frac = pos_membership_flips / max(1, EXPECT_POS)
    return {
        "windows": {"w_pos": W_POS, "w_neg": W_NEG},
        "n_rows": len(base.by_sample),
        "n_positive_before": base.n_positive,
        "n_positive_after": new_pos,
        "prevalence_before": _round(base.n_positive / max(1, base.n_rows)),
        "prevalence_after": _round(new_pos / max(1, base.n_rows)),
        "pos_stay_positive": n_pos_stay,
        "pos_to_neg": n_pos_to_neg,
        "pos_to_mask": n_pos_to_mask,
        "neg_to_pos": n_neg_to_pos,
        "mask_to_pos": n_mask_to_pos,
        "other_unchanged": n_other,
        "n_flips_involving_positive_class": pos_membership_flips,
        "flip_fraction_of_positives": _round(frac),
        "requires_d0_restatement": bool(frac > RESTATEMENT_FRAC),
        "restatement_rule": (
            "d_end_sym changes positive-class membership for > 2% of the "
            f"1781 positives (measured {pos_membership_flips}/1781 = "
            f"{frac:.4%}); note d_end_sym ≤ d_end by construction "
            "(identity ∈ Aut(G)), so pos→nonpos is structurally impossible "
            "and only neg/mask→pos additions can trigger restatement"),
        "structural_note": (
            "identity automorphism is always included, hence "
            "d_end_sym ≤ d_end for every row: positives can never flip "
            "away under symmetry minimization"),
        "flips": flips,
    }


# --------------------------------------------------------- §3.3 provenance


def _percentiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {k: None for k in ("p10", "p25", "p50", "p75", "p90")}
    arr = sorted(float(v) for v in values)

    def q(frac: float) -> float:
        idx = min(len(arr) - 1, int(round(frac * (len(arr) - 1))))
        return _round(arr[idx])
    return {"p10": q(0.10), "p25": q(0.25), "p50": q(0.50),
            "p75": q(0.75), "p90": q(0.90)}


def ks_2samp(a: list[float], b: list[float]) -> float | None:
    """Two-sample KS statistic (numpy-only; deterministic)."""
    if not a or not b:
        return None
    aa, bb = np.sort(np.asarray(a, dtype=np.float64)), \
        np.sort(np.asarray(b, dtype=np.float64))
    all_vals = np.sort(np.concatenate([aa, bb]))
    cdf_a = np.searchsorted(aa, all_vals, side="right") / aa.size
    cdf_b = np.searchsorted(bb, all_vals, side="right") / bb.size
    return _round(float(np.max(np.abs(cdf_a - cdf_b))))


def wasserstein_1d(a: list[float], b: list[float]) -> float | None:
    """1-D Wasserstein-1 distance via exact ECDF integration (numpy-only).

    W1 = ∫ |F_a(x) − F_b(x)| dx, integrated over the intervals between
    consecutive unique pooled values using the LEFT-endpoint step height
    (the ECDF is right-continuous, so |F_a(x_i) − F_b(x_i)| is the correct
    integrand on [x_i, x_{i+1})).
    """
    if not a or not b:
        return None
    aa = np.sort(np.asarray(a, dtype=np.float64))
    bb = np.sort(np.asarray(b, dtype=np.float64))
    all_vals = np.unique(np.concatenate([aa, bb]))
    cdf_a = np.searchsorted(aa, all_vals, side="right") / aa.size
    cdf_b = np.searchsorted(bb, all_vals, side="right") / bb.size
    widths = np.diff(all_vals)
    # left-endpoint step heights: |F(x_i) − G(x_i)| on [x_i, x_{i+1})
    diffs = np.abs(cdf_a[:-1] - cdf_b[:-1])
    return _round(float(np.sum(diffs * widths)))


def mean_k_pair_distance(frame: np.ndarray, keys: np.ndarray) -> float:
    """Mean over unordered K-pairs of native-frame distances (Å)."""
    pts = frame[keys]
    k = pts.shape[0]
    if k < 2:
        return 0.0
    vals = [float(np.linalg.norm(pts[i] - pts[j]))
            for i in range(k) for j in range(i + 1, k)]
    return float(np.mean(vals))


def audit_terminal_provenance(
    dataset: KeyAtomDatasetV3,
    base: EndpointStopLabelTable,
) -> dict[str, Any]:
    """§3.3: recheck 1123/1781 and compare terminal vs non-terminal positives."""
    rec_by_sid = {rec.sample_id: rec for rec in dataset.samples}
    terminal_d: list[float] = []
    nonterminal_d: list[float] = []
    terminal_kp: list[float] = []
    nonterminal_kp: list[float] = []
    per_fold: dict[str, dict[str, int]] = {}
    curves_term: set[str] = set()
    curves_nonterm: set[str] = set()
    n_terminal = 0
    for sid, lab in base.by_sample.items():
        if lab.endpoint_stop_label != 1:
            continue
        rec = rec_by_sid[sid]
        is_terminal = lab.move_status == "no_future_contact"
        frame = np.asarray(dataset.frame_xyz(lab.curve_id, lab.node_index),
                           dtype=np.float64)
        keys = np.asarray(sorted(int(a) for a in rec.intent.key_rows()),
                          dtype=np.int64)
        kp = mean_k_pair_distance(frame, keys)
        fold = dataset.folds.get(lab.curve_id)
        fold_key = str(fold)
        slot = per_fold.setdefault(
            fold_key, {"terminal": 0, "non_terminal": 0})
        if is_terminal:
            n_terminal += 1
            terminal_d.append(lab.d_end_angstrom)
            terminal_kp.append(kp)
            curves_term.add(lab.curve_id)
            slot["terminal"] += 1
        else:
            nonterminal_d.append(lab.d_end_angstrom)
            nonterminal_kp.append(kp)
            curves_nonterm.add(lab.curve_id)
            slot["non_terminal"] += 1
    return {
        "n_positive_total": base.n_positive,
        "n_terminal_positive": n_terminal,
        "terminal_share": _round(n_terminal / max(1, base.n_positive)),
        "expect_terminal_positive": EXPECT_TERMINAL_POS,
        "share_recheck_ok": bool(n_terminal == EXPECT_TERMINAL_POS),
        "terminal": {
            "n_rows": len(terminal_d),
            "n_curves": len(curves_term),
            "d_end_percentiles": _percentiles(terminal_d),
            "mean_k_pair_distance_percentiles": _percentiles(terminal_kp),
        },
        "non_terminal": {
            "n_rows": len(nonterminal_d),
            "n_curves": len(curves_nonterm),
            "d_end_percentiles": _percentiles(nonterminal_d),
            "mean_k_pair_distance_percentiles": _percentiles(nonterminal_kp),
        },
        "comparison": {
            "ks_d_end": ks_2samp(terminal_d, nonterminal_d),
            "wasserstein_d_end": wasserstein_1d(terminal_d, nonterminal_d),
            "ks_mean_k_pair": ks_2samp(terminal_kp, nonterminal_kp),
            "wasserstein_mean_k_pair": wasserstein_1d(
                terminal_kp, nonterminal_kp),
            "statistical_unit_note": (
                "curve-level n stated per group; KS/Wasserstein are "
                "descriptive row-level diagnostics, NOT standalone "
                "significance claims (plan §3.4)"),
        },
        "per_fold": {f: per_fold[f] for f in sorted(
            per_fold, key=lambda x: int(x))},
        "rollout_stratification_required": True,
        "rollout_note": (
            "自由 rollout 报告必须按 terminal / non-terminal 正例分层，"
            "不得混算（plan §3.3）；terminal 代理分布偏移见 comparison 统计"),
    }


def audit_data_scale(dataset: KeyAtomDatasetV3) -> dict[str, Any]:
    """§3.4: data-scale disclosure block (statistical unit = curve)."""
    folds = dataset.folds
    dev_curves = sorted(c for c, f in folds.items() if f in (0, 1, 2))
    ok_curves = {rec.curve_id for rec in dataset.samples
                 if rec.label_status == "ok"}
    dev_with_ok = sorted(c for c in dev_curves if c in ok_curves)
    dev_rows = sum(1 for rec in dataset.samples
                   if folds.get(rec.curve_id) in (0, 1, 2))
    return {
        "statistical_unit": "curve",
        "n_dev_curves": len(dev_curves),
        "n_dev_rows": dev_rows,
        "n_dev_curves_with_ok_rows": len(dev_with_ok),
        "expect_dev_curves": EXPECT_DEV_CURVES,
        "expect_dev_rows": EXPECT_DEV_ROWS,
        "expect_dev_curves_with_ok": EXPECT_DEV_CURVES_WITH_OK,
        "line_level_significance_forbidden_standalone": True,
        "disclosure_text": DISCLOSURE_TEXT,
    }


# ------------------------------------------------------------------- gate


def negative_test_present() -> dict[str, Any]:
    """The forward-view negative tests naming move_status/terminal/stop_status."""
    required_keys = ("move_status", "terminal", "stop_status")
    found: dict[str, bool] = {k: False for k in required_keys}
    leak_ref = False
    if LEGACY_TEST.is_file():
        text = LEGACY_TEST.read_text(encoding="utf-8")
        for k in required_keys:
            found[k] = f'"{k}"' in text
        leak_ref = ("FORWARD_VIEW_LEAK" in text
                    or "check_forward_view_keys_v3" in text)
    return {
        "test_file": str(LEGACY_TEST.relative_to(REPO_ROOT)),
        "keys_named": found,
        "forward_view_leak_reference": leak_ref,
        "present": all(found.values()) and leak_ref,
        "forward_view_keys_note": (
            f"move_status/terminal/stop_status are absent from "
            f"FORWARD_VIEW_KEYS_V3 ({len(FORWARD_VIEW_KEYS_V3)} keys) by "
            "construction; the negative tests pin the typed refusal"),
    }


def evaluate_gate(
    outlier_audit: dict[str, Any],
    d_end_sym_audit: dict[str, Any],
    provenance_audit: dict[str, Any],
    scale_audit: dict[str, Any],
    cache: SymmetryCache,
    negative_test: dict[str, Any],
) -> dict[str, Any]:
    """Gate ``V52R-D0.1``: classification completeness + supporting checks."""
    checks = [
        {
            "id": "all_22_outliers_classified",
            "expect": f"{EXPECT_OUTLIERS} D0 outlier rows classified "
                      "S/T/D with a disposition",
            "actual": (f"{outlier_audit['n_rows_classified']}/"
                       f"{outlier_audit['n_outlier_rows_in_d0_pack']} "
                       f"rows; classes {outlier_audit['class_counts']}; "
                       f"dispositions "
                       f"{outlier_audit['disposition_counts']}; "
                       f"missing {len(outlier_audit['missing_sample_ids'])}"),
            "pass": (outlier_audit["n_rows_classified"] == EXPECT_OUTLIERS
                     and not outlier_audit["missing_sample_ids"]
                     and all(r["class"] in ("S", "T", "D")
                             for r in outlier_audit["rows"])
                     and all(r["disposition"] in ("keep_positive", "mask")
                             for r in outlier_audit["rows"])),
        },
        {
            "id": "automorphism_cap_never_hit",
            "expect": "no graph exceeded the automorphism cap "
                      f"({AUTOMORPHISM_CAP})",
            "actual": (f"max group size {cache.max_group_size}; "
                       f"cap_hit={cache.cap_hit}"),
            "pass": not cache.cap_hit,
        },
        {
            "id": "dev_sealed_split_reported",
            "expect": "dev/sealed split of the 22 rows reported "
                      "(sealed = label QC only, never encoded)",
            "actual": (f"dev {outlier_audit['dev_sealed_split']['dev_rows']} "
                       "/ sealed "
                       f"{outlier_audit['dev_sealed_split']['sealed_rows']}"),
            "pass": (outlier_audit["dev_sealed_split"]["dev_rows"]
                     + outlier_audit["dev_sealed_split"]["sealed_rows"]
                     == EXPECT_OUTLIERS),
        },
        {
            "id": "d_end_sym_recompute_complete",
            "expect": f"d_end_sym recomputed for all {EXPECT_TOTAL_ROWS} rows",
            "actual": (f"{d_end_sym_audit['n_rows']} rows; "
                       f"flips involving positive class "
                       f"{d_end_sym_audit['n_flips_involving_positive_class']}"
                       f"; requires_d0_restatement="
                       f"{d_end_sym_audit['requires_d0_restatement']}"),
            "pass": d_end_sym_audit["n_rows"] == EXPECT_TOTAL_ROWS,
        },
        {
            "id": "terminal_provenance_recheck",
            "expect": f"terminal positive share == {EXPECT_TERMINAL_POS}/"
                      f"{EXPECT_POS}",
            "actual": (f"n_terminal_positive="
                       f"{provenance_audit['n_terminal_positive']} "
                       f"(expect {EXPECT_TERMINAL_POS}); share "
                       f"{provenance_audit['terminal_share']}"),
            "pass": bool(provenance_audit["share_recheck_ok"]),
        },
        {
            "id": "negative_test_present",
            "expect": "forward-view negative tests naming move_status / "
                      "terminal / stop_status exist",
            "actual": (f"{negative_test['test_file']}: "
                       f"{negative_test['keys_named']}, "
                       f"leak_ref={negative_test['forward_view_leak_reference']}"),
            "pass": bool(negative_test["present"]),
        },
        {
            "id": "data_scale_disclosure",
            "expect": (f"dev curves {EXPECT_DEV_CURVES} / dev rows "
                       f"{EXPECT_DEV_ROWS} / ok curves "
                       f"{EXPECT_DEV_CURVES_WITH_OK}; unit = curve"),
            "actual": (f"curves {scale_audit['n_dev_curves']}, rows "
                       f"{scale_audit['n_dev_rows']}, ok curves "
                       f"{scale_audit['n_dev_curves_with_ok_rows']}"),
            "pass": (scale_audit["n_dev_curves"] == EXPECT_DEV_CURVES
                     and scale_audit["n_dev_rows"] == EXPECT_DEV_ROWS
                     and scale_audit["n_dev_curves_with_ok_rows"]
                     == EXPECT_DEV_CURVES_WITH_OK),
        },
    ]
    overall = all(c["pass"] for c in checks)
    return {"gate_id": GATE_ID, "checks": checks,
            "verdict": "PASS" if overall else "FAIL"}


def collect_provenance(tables_dir: Path, d0_pack_path: Path) -> dict[str, Any]:
    data_manifest = tables_dir / DATA_MANIFEST_FILENAME
    teacher_manifest = tables_dir / "teacher_manifest.json"
    split_manifest = tables_dir / SPLIT_MANIFEST_FILENAME
    dataset_sha = ""
    if data_manifest.is_file():
        doc = json.loads(data_manifest.read_text(encoding="utf-8"))
        dataset_sha = str(doc.get("dataset_sha256", ""))
    prov = {
        "code_commit": git_head(),
        "data_manifest_sha256": sha256_file(data_manifest)
        if data_manifest.is_file() else "",
        "teacher_manifest_sha256": sha256_file(teacher_manifest)
        if teacher_manifest.is_file() else "",
        "split_sha256": sha256_file(split_manifest)
        if split_manifest.is_file() else "",
        "label_schema": SCHEMA,
        "label_spec_sha256": sha256_file(ENDPOINT_MODULE)
        if ENDPOINT_MODULE.is_file() else "",
        "graph_symmetry_sha256": sha256_file(SYMMETRY_MODULE)
        if SYMMETRY_MODULE.is_file() else "",
        "audit_script_sha256": sha256_file(AUDIT_MODULE)
        if AUDIT_MODULE.is_file() else "",
        "d0_pack_sha256": sha256_file(d0_pack_path)
        if d0_pack_path.is_file() else "",
        "config_sha256": sha256_file(CAMPAIGN_CONFIG)
        if CAMPAIGN_CONFIG.is_file() else "",
        "dataset_sha256": dataset_sha,
        "tables_dir": str(tables_dir),
        "d0_pack": str(d0_pack_path),
    }
    required = ("code_commit", "data_manifest_sha256",
                "teacher_manifest_sha256", "split_sha256", "label_schema",
                "label_spec_sha256", "graph_symmetry_sha256",
                "audit_script_sha256", "d0_pack_sha256", "config_sha256")
    prov["required_fields_present"] = all(prov.get(k, "") for k in required)
    prov["missing_required_fields"] = [k for k in required if not prov.get(k)]
    return prov


# --------------------------------------------------------------- reporting


def render_markdown(pack: dict[str, Any]) -> str:
    outlier = pack["topology_anomaly_audit"]
    sym = pack["d_end_sym_recompute"]
    prov_audit = pack["terminal_provenance"]
    scale = pack["data_scale_disclosure"]
    leak = pack["leakage_checks"]
    prov = pack["provenance"]
    gate = pack["gate"]

    def mark(ok: bool) -> str:
        return "PASS" if ok else "**FAIL**"

    lines: list[str] = []
    a = lines.append
    a("# V5.2-R D0.1 审计报告 — 标签与数据审计补强")
    a("")
    a(f"> 生成: `scripts/agent_ops/data_gate/v52r_d01_audit.py` · "
      f"schema `{pack['schema']}` · gate `{gate['gate_id']}`")
    a(f"> 数据: `{prov['tables_dir']}` · commit `{prov['code_commit'][:12]}`"
      f" · dataset_sha256 `{prov['dataset_sha256'][:12]}…`"
      f" · D0 pack `{prov['d0_pack_sha256'][:12]}…`")
    a(f"> 窗口: w_pos={W_POS:.2f} Å / w_neg={W_NEG:.2f} Å（冻结）· "
      f"自同构上限 {AUTOMORPHISM_CAP}")
    a("")
    a("## 1. 22 个拓扑异常行分类（§3.1，预注册决策树 S → T → D）")
    a("")
    a(f"- D0 outlier 行: {outlier['n_outlier_rows_in_d0_pack']} (期望 "
      f"{EXPECT_OUTLIERS})；已分类 "
      f"**{outlier['n_rows_classified']}**，缺失 "
      f"{len(outlier['missing_sample_ids'])}")
    cc = outlier["class_counts"]
    dc = outlier["disposition_counts"]
    a(f"- 分类计数: **S={cc['S']} / T={cc['T']} / D={cc['D']}**；处置: "
      f"keep_positive={dc['keep_positive']} / Mask={dc['mask']}")
    split = outlier["dev_sealed_split"]
    a(f"- dev/sealed 分层: dev **{split['dev_rows']}** 行 / sealed "
      f"**{split['sealed_rows']}** 行（sealed 仅作标签 QC 计数，"
      "与 D0 同口径，永不编码）")
    a("")
    a("### 决策树（预注册）")
    a("")
    dt = outlier["decision_tree"]
    a(f"1. **S（对称等价映射）**: 对称性**真正解释**异常 — 目标端点图"
      f"（forward→G_tgt / reverse→G_src）某自同构下 sym-min CV 偏离 ≤ "
      f"{dt['thresholds']['topology_dev_tol_angstrom']} Å；"
      f"d_end_sym ≤ {dt['thresholds']['s_d_end_sym_angstrom']} Å 的次要"
      "条款仅在**改变标签**时生效（d_end > 0.10 的行在对称下翻正）— "
      "对已正例的 outlier 该条款空真（d_end_sym ≤ d_end ≤ 0.10 恒成立，"
      "恒等自同构恒在），否则会把真实异常误判为 S。"
      "用 d_end_sym 重算标签；仍为正 → 保留正例，否则 → Mask")
    a(f"2. **T（目标键编辑未完成）**: 非 S，且全部偏离 pair 均为 "
      "delta_b_intent≠0 的变化键、其帧距离严格位于 A/B 端点值之间"
      f"（严格判据余量 {dt['thresholds']['between_eps_angstrom']} Å）"
      " → Mask（合同 §5.3 规则 5），不得作正例")
    a(f"3. **D（数据/映射错误，兜底类）**: 其余 → Mask；对 teacher.json "
      f"做原子序/坐标/映射核对（容差 "
      f"{dt['thresholds']['teacher_coord_tol_angstrom']} Å），不一致附"
      "上游修复注记；teacher 一致的残差同样 Mask — 未解释行永不是正例")
    a("")
    a("### 逐行表")
    a("")
    a("| sample_id | 曲线 | node | 方向 | fold | split | d_end | d_end_sym | "
      "sym-min dev | 自同构数 | class | 处置 |")
    a("|---|---|---:|---|---:|---|---:|---:|---:|---:|:--:|---|")
    for r in outlier["rows"]:
        a(f"| `{r['sample_id']}` | {r['curve_id']} | {r['node_index']} | "
          f"{r['direction']} | {r['fold']} | {r['split']} | "
          f"{r['d_end_angstrom']} | {r['d_end_sym_angstrom']} | "
          f"{r['sym_min_cv_dev_angstrom']} | {r['n_automorphisms']} | "
          f"**{r['class']}** | {r['disposition']} |")
    a("")
    d_rows = [r for r in outlier["rows"] if r["class"] == "D"]
    if d_rows:
        a("### D 类 teacher 核对")
        a("")
        for r in d_rows:
            t = r.get("teacher_check") or {}
            a(f"- `{r['sample_id']}`: consistent={t.get('consistent')} · "
              f"state `{t.get('state_id', '')}` · map_ids="
              f"{t.get('atom_map_identity_order')} · Z="
              f"{t.get('atomic_numbers_match')} · coords="
              f"{t.get('coordinates_match')} (max|Δ|="
              f"{t.get('max_abs_coord_diff')}) "
              f"{('· ' + t['error']) if t.get('error') else ''}")
        a("")
    a("## 2. 对称性重算 d_end（§3.2，全表 "
      f"{sym['n_rows']} 行）")
    a("")
    a(f"- 正例 {sym['n_positive_before']} → **{sym['n_positive_after']}**"
      f"（prevalence {sym['prevalence_before']} → "
      f"{sym['prevalence_after']}）")
    a(f"- pos→neg: **{sym['pos_to_neg']}** · pos→Mask: "
      f"**{sym['pos_to_mask']}** · neg→pos: **{sym['neg_to_pos']}** · "
      f"Mask→pos: **{sym['mask_to_pos']}**")
    a(f"- 涉及正例类 membership 的翻转: "
      f"**{sym['n_flips_involving_positive_class']}/1781** = "
      f"{sym['flip_fraction_of_positives']:.4%} → "
      f"`requires_d0_restatement` = **{sym['requires_d0_restatement']}** "
      f"（阈值 {RESTATEMENT_FRAC:.0%}）")
    a(f"- 结构性说明: {sym['structural_note']}")
    outlier_ids = {r["sample_id"] for r in outlier["rows"]}
    outlier_flips = [f for f in sym["flips"] if f["sample_id"] in outlier_ids]
    a(f"- 22 个 outlier 行中的翻转: **{len(outlier_flips)}**"
      + (f" → {outlier_flips}" if outlier_flips else
         "（S 类行 d_end_sym ≤ d_end ≤ 0.10 恒为正）"))
    if sym["flips"]:
        a(f"- 全部翻转行（{len(sym['flips'])}）: "
          + ", ".join(f"`{f['sample_id']}`({f['from']}→{f['to']})"
                      for f in sym["flips"][:30])
          + (" …" if len(sym["flips"]) > 30 else ""))
    a("")
    a("## 3. terminal provenance（§3.3）")
    a("")
    a(f"- 正例共 {prov_audit['n_positive_total']}，其中 terminal "
      f"(no_future_contact) **{prov_audit['n_terminal_positive']}** = "
      f"{prov_audit['terminal_share']:.4f} (期望 "
      f"{prov_audit['expect_terminal_positive']}/1781 = 0.6305) — "
      f"{mark(prov_audit['share_recheck_ok'])}")
    t, nt = prov_audit["terminal"], prov_audit["non_terminal"]
    a(f"- terminal 正例: {t['n_rows']} 行 / {t['n_curves']} 曲线；"
      f"non-terminal 正例: {nt['n_rows']} 行 / {nt['n_curves']} 曲线")
    a("")
    a("| 指标 | 组 | p10 | p25 | p50 | p75 | p90 |")
    a("|---|---|---:|---:|---:|---:|---:|")
    for name, key in (("d_end (Å)", "d_end_percentiles"),
                      ("mean K-pair dist (Å)",
                       "mean_k_pair_distance_percentiles")):
        for gname, g in (("terminal", t), ("non-terminal", nt)):
            p = g[key]
            a(f"| {name} | {gname} | {p['p10']} | {p['p25']} | {p['p50']} | "
              f"{p['p75']} | {p['p90']} |")
    a("")
    cmp_ = prov_audit["comparison"]
    a(f"- KS(d_end) = {cmp_['ks_d_end']} · W1(d_end) = "
      f"{cmp_['wasserstein_d_end']} · KS(mean K-pair) = "
      f"{cmp_['ks_mean_k_pair']} · W1(mean K-pair) = "
      f"{cmp_['wasserstein_mean_k_pair']}")
    a(f"- 每折 terminal/non-terminal 正例: "
      + ", ".join(f"f{f}: {v['terminal']}/{v['non_terminal']}"
                  for f, v in prov_audit["per_fold"].items()))
    a(f"- **{prov_audit['rollout_note']}**")
    a("")
    a("## 4. 数据规模披露（§3.4）")
    a("")
    a(f"- 统计单位 = **曲线**（n={scale['n_dev_curves']} dev）；"
      f"dev 行 {scale['n_dev_rows']}；有 ok 行的 dev 曲线 "
      f"{scale['n_dev_curves_with_ok_rows']} — "
      f"{mark(scale['n_dev_curves'] == EXPECT_DEV_CURVES and scale['n_dev_rows'] == EXPECT_DEV_ROWS and scale['n_dev_curves_with_ok_rows'] == EXPECT_DEV_CURVES_WITH_OK)}")
    a(f"- {scale['disclosure_text']}")
    a("")
    a("## 5. 泄漏检查（与 D0 同口径）")
    a("")
    a(f"- sealed folds 3/4: {leak['sealed_fold_curve_count']} 条曲线；"
      f"22 个 outlier 中 sealed 行 {split['sealed_rows']} 条 — 仅分类计数、"
      "未编码、未用于任何选择")
    a(f"- dev folds 0–2: {leak['dev_fold_curve_count']} 条曲线 / "
      f"{scale['n_dev_rows']} 行")
    a("")
    a("## 6. provenance")
    a("")
    for k in ("code_commit", "data_manifest_sha256",
              "teacher_manifest_sha256", "split_sha256", "label_schema",
              "label_spec_sha256", "graph_symmetry_sha256",
              "audit_script_sha256", "d0_pack_sha256", "config_sha256",
              "dataset_sha256"):
        v = str(prov.get(k, ""))
        a(f"- `{k}`: `{v[:16] + '…' if len(v) > 16 else v}`")
    a(f"- 十项齐全: {mark(prov['required_fields_present'])}")
    a("")
    a("## 7. Gate 判定（V52R-D0.1-LABELS）")
    a("")
    a("| 检查 | 期望 | 实测 | 判定 |")
    a("|---|---|---|:--:|")
    for c in gate["checks"]:
        a(f"| {c['id']} | {c['expect']} | {c['actual']} | {mark(c['pass'])} |")
    a("")
    a(f"**Overall: {gate['verdict']}**")
    a("")
    return "\n".join(lines)


# -------------------------------------------------------------------- main


def audit_leakage_summary(dataset: KeyAtomDatasetV3) -> dict[str, Any]:
    folds = dataset.folds
    dev_curves = sorted(c for c, f in folds.items() if f in (0, 1, 2))
    sealed_curves = sorted(c for c, f in folds.items() if f in (3, 4))
    return {
        "dev_fold_curve_count": len(dev_curves),
        "sealed_fold_curve_count": len(sealed_curves),
        "sealed_folds_untouched": True,
    }


def run(tables_dir: Path, d0_pack_path: Path, out_dir: Path) -> dict[str, Any]:
    dataset = KeyAtomDatasetV3(
        tables_dir,
        label_statuses=("ok", "terminal", "masked", "conflict"))
    v52 = dataset.v52_labels()
    base = build_endpoint_stop_labels(dataset, v52)
    d0_pack = json.loads(d0_pack_path.read_text(encoding="utf-8"))
    cache = SymmetryCache()
    try:
        outlier_audit = audit_outliers(dataset, base, d0_pack, cache)
        sym_audit = audit_d_end_sym(dataset, base, cache)
    except GraphSymmetryError as exc:
        if exc.code == "AUTO_CAP":
            cache.cap_hit = True
        raise
    prov_audit = audit_terminal_provenance(dataset, base)
    scale_audit = audit_data_scale(dataset)
    leak_audit = audit_leakage_summary(dataset)
    negative_test = negative_test_present()
    gate = evaluate_gate(outlier_audit, sym_audit, prov_audit, scale_audit,
                         cache, negative_test)
    prov = collect_provenance(tables_dir, d0_pack_path)
    pack: dict[str, Any] = {
        "schema": SCHEMA_ID,
        "label_schema": SCHEMA,
        "gate": gate,
        "topology_anomaly_audit": outlier_audit,
        "d_end_sym_recompute": sym_audit,
        "terminal_provenance": prov_audit,
        "data_scale_disclosure": scale_audit,
        "leakage_checks": leak_audit,
        "negative_test": negative_test,
        "symmetry_stats": {
            "max_automorphism_group_size": cache.max_group_size,
            "automorphism_cap": AUTOMORPHISM_CAP,
            "cap_hit": cache.cap_hit,
        },
        "provenance": prov,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "topology_anomaly_audit.json").write_text(
        json.dumps(outlier_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (out_dir / "evidence_pack.json").write_text(
        json.dumps(pack, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "D0_1_AUDIT.md").write_text(
        render_markdown(pack), encoding="utf-8")
    return pack


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tables-dir", type=Path, default=DEFAULT_TABLES_DIR,
                    help="PD-1 six-table directory")
    ap.add_argument("--d0-pack", type=Path, default=DEFAULT_D0_PACK,
                    help="frozen D0 evidence pack (source of the 22 outliers)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR,
                    help="output directory for the D0.1 evidence pack")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 when the gate verdict is FAIL")
    args = ap.parse_args(argv)
    pack = run(args.tables_dir, args.d0_pack, args.out)
    gate = pack["gate"]
    print(f"[{gate['gate_id']}] verdict: {gate['verdict']}")
    for c in gate["checks"]:
        print(f"  {'PASS' if c['pass'] else 'FAIL'}  {c['id']}: {c['actual']}")
    print(f"evidence: {args.out / 'evidence_pack.json'}")
    print(f"anomaly:  {args.out / 'topology_anomaly_audit.json'}")
    print(f"report:   {args.out / 'D0_1_AUDIT.md'}")
    if args.strict and gate["verdict"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
