#!/usr/bin/env python3
"""V5.3-MP S0-B: authorized IRC extraction + teacher build for the frozen cohort.

Tier-agnostic by construction: the build order is the ``--cohort`` file's
``pool_candidates`` (26k for the frozen 20k build; 65k for a 50k build —
``r3_v53mp_freeze_cohort.py --pool-candidate-target 65000``); there is no
tier cap in this script.  Builds the V5.3-MP six-table dataset (schema
``r3_6table_v2``, identical
layout to the 30h R3 build) directly from the quarantined B3LYP-D3/TZVP IRC
archive, the G1 P1 mappings and the (audited) TS table — the
``outputs/pesn/dual_ts_v2_20261015/wp2_fit50`` JSON lineage covers only the
50-curve cohort, so the cohort build extracts frames from the source archive
through the audited, scope-checked pesn truth bridge
(:func:`pes2ts_core.pesn.offline.data.truth_source.stream_scoped_irc_frames`
— every reaction read is appended to ``data/manifests/truth_access_log.jsonl``
by the accessor; this build's slice is copied verbatim into the output dir).

Truth-access discipline (plan §4.2, hard gate = no UNAUTHORIZED access):

* scope is enforced BEFORE the archive is opened: every requested reaction
  must sit in the frozen predictor split's train-side subsets
  (train_fit/train_dev/train_calibration of ``outputs/pesn/g3_p1/splits/
  pesn_irc_split.json``); confirm-listed reactions are never requested;
* one bulk audited TS-table read (charge/multiplicity/atomic_numbers only);
* per-reaction frame reads happen only inside the audited bridge stream.

IRC layout (verified 2026-10-21/22 on the quarantined archive): frame 0 is
the TS (``irc_index.ts_index == 0`` archive-wide; matches ``ts.parquet``
where the two share a rigid frame); frames ``1..brk`` are the first IRC
direction, frame ``brk+1`` is a second TS copy (TS'), frames ``brk+2..`` the
second direction.  The canonical curve is ``[seg_R reversed] + [seg_P[1:]]``
with ONE shared TS (frame 0), mirroring the R3 ``Curve`` construction;
consecutive-frame continuity of the canonical curve is gated with the frozen
native thresholds (2 deg / 0.1 A) — failures are masked
(``gauge_unresolved_excluded``), never silently re-framed.

Orientation (which segment end is R vs P) is decided per reaction from
frame-level bond evidence against the P1 endpoint graphs (exclusive-bond
score, with an edit-direction comparative fallback), cross-checked against
the G1 ``orientation`` field; evidence and any disagreement are recorded in
the build ledger.

Subcommands::

    stage     # audited streaming pass -> staging npz chunks + extraction manifest
    build     # parallel per-chunk curve build -> CSV fragments (no truth reads)
    assemble  # concat fragments -> six tables + manifests + build ledger
    all       # stage + build + assemble

Usage::

    python3 scripts/agent_ops/data_gate/r3_v53mp_build_tables_h5.py all \
        --cohort outputs/pesn/v53mp/cohort/v53mp_cohort_v1.json \
        --out outputs/pesn/v53mp/data [--workers 24] [--limit N] [--force]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import io
import json
import math
import os
import shutil
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_r3_builder():
    """Load the frozen R3 builder module (Curve / teacher / audit helpers)."""
    path = REPO_ROOT / "scripts/agent_ops/data_gate/r3_build_tables.py"
    spec = importlib.util.spec_from_file_location("r3_build_tables_lib", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


r3b = _load_r3_builder()

from pes2ts_core.g0.truth import truth_reader  # noqa: E402
from pes2ts_core.pesn.contracts.key_atom import build_key_atom_task  # noqa: E402
from pes2ts_core.pesn.offline.data import truth_source  # noqa: E402
from pes2ts_core.pesn.offline.data.key_atom_teacher import (  # noqa: E402
    MAIN_D0,
    TEACHER_CAP_ANGSTROM,
    verify_frame_continuity,
)

TOOL_VERSION = "pes2ts_r3_v53mp_build_tables_h5_v1"
SCHEMA_VERSION = r3b.SCHEMA_VERSION          # r3_6table_v2
INTENT_VERSION = r3b.INTENT_VERSION
TEACHER_VERSION = r3b.TEACHER_VERSION
CANONICAL_UNITS = "angstrom"

G3P1_SPLIT = "outputs/pesn/g3_p1/splits/pesn_irc_split.json"
P1_REL = "data/interim/g1_truth/p1_mapping"
MANIFESTS_DIR = "data/manifests"
TS_PARQUET_VIA = "truth_reader.load_ts_table (bulk audited read; " \
                 "charge/multiplicity/atomic_numbers only)"

CHUNK_SIZE = 200
#: minimum per-direction frame count (TS + 2 steps) to form a usable branch
MIN_SEGMENT_FRAMES = 3
#: TS-energy sanity: E(TS) must be within this of the trajectory max (Eh)
TS_ENERGY_MAX_TOL_EH = 5.0e-3
#: orientation decision margin (exclusive-bond score difference)
ORIENTATION_MARGIN = 0.5
#: comparative (edit-direction) fallback margin: >= 2/3 of edits must agree
ORIENTATION_COMPARATIVE_MARGIN = 1.0 / 3.0

TABLES_ORDER = (
    "irc_curve.csv",
    "irc_node.csv",
    "irc_edge.csv",
    "reaction_intent.csv",
    "curve_quality.csv",
    "sample_view.csv",
    "branch_diagnostics.csv",
)

TABLE_HEADERS = {
    "irc_curve.csv": ["curve_id", "canonical_irc_id", "n_nodes", "n_edges",
                      "ts_index", "endpoint_a_index", "endpoint_b_index",
                      "source_path", "energy_a", "energy_b"],
    "irc_node.csv": ["curve_id", "node_index", "x", "y", "z", "progress",
                     "is_ts", "is_endpoint", "energy", "gradient_norm",
                     "quality", "atomic_numbers", "coords_flat",
                     "coords_shape"],
    "irc_edge.csv": ["curve_id", "intent_id", "edge_index", "src_node_index",
                     "dst_node_index", "delta_xyz_flat", "delta_xyz_shape",
                     "delta_xyz_offsets", "atom_map_sha256",
                     "delta_xyz_sha256", "units", "same_ref_frame"],
    "reaction_intent.csv": ["intent_id", "curve_id", "direction",
                            "reverse_of_intent_id", "intent_pair_id",
                            "intent_underdetermined", "g_src", "g_tgt",
                            "delta_b_intent", "cv_spec", "intent_role",
                            "rules", "intent_version"],
    "curve_quality.csv": ["curve_id", "quality_status", "quality_score",
                          "segment_quality", "branch_quality", "note"],
    "sample_view.csv": ["sample_id", "curve_id", "node_index", "intent_id",
                        "d0", "x", "y", "z", "label_weight", "curve_weight",
                        "sample_weight", "label_status", "feature_sha",
                        "label_sha", "sensitivity_only", "gauge_ref",
                        "target_node_index", "target_alpha",
                        "target_action_u", "target_progress_q", "stop_label",
                        "mask", "fold", "disposition", "coords_flat",
                        "coords_shape"],
    "branch_diagnostics.csv": ["curve_id", "branch_id", "endpoint_label",
                               "endpoint_side", "n_frames", "v5_gate_ok",
                               "v5_continuity_passed", "v5_frame_layer",
                               "v5_unresolved", "consec_rot_deg",
                               "consec_trans_angstrom", "step_max_angstrom",
                               "source_path"],
}


class BuildError(RuntimeError):
    """Per-reaction exclusion (typed reason; recorded in the ledger)."""

    def __init__(self, reason: str, detail: str = "") -> None:
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# per-curve construction (worker side)
# ---------------------------------------------------------------------------


class H5Bundle:
    """Duck-typed sibling of ``r3b.ReactionBundle`` for the raw-IRC lineage.

    Exposes exactly the attributes ``r3b.Curve`` and the row-emitting code
    consume: ``task`` (KeyAtomTask from a synthetic wp2 task document),
    ``branch_frames``/``branch_label``/``branch_ep0``/``branch_ep1`` (two
    branches, endpoint -> TS, the shared canonical TS as last frame),
    ``atomic_numbers``/``atom_map_ids``/``charge``/``multiplicity`` and
    :meth:`exclusive_bond_score` (row-space, same semantics as the wp2 path).
    """

    def __init__(self, rid: str, task_doc: dict, p1_doc: dict,
                 p1_path: str, branch_frames: Dict[str, np.ndarray],
                 branch_label: Dict[str, str],
                 atomic_numbers: Sequence[int], charge: int,
                 multiplicity: int) -> None:
        self.rid = rid
        self.task_doc = task_doc
        self.p1_doc = p1_doc
        self.p1_path = p1_path
        self.task = build_key_atom_task(task_doc, p1_doc, p1_path=str(p1_path))
        self.branch_frames = branch_frames
        self.branch_label = branch_label
        by_label = {lab: bid for bid, lab in branch_label.items()}
        if set(by_label) != {"A", "B"}:
            raise BuildError("branch_labels", f"{rid}: {by_label}")
        self.branch_ep0 = by_label["A"]   # A starts at endpoints[0] (= R slot)
        self.branch_ep1 = by_label["B"]
        self.atomic_numbers = [int(z) for z in atomic_numbers]
        self.atom_map_ids = [
            f"m{i + 1:04d}" for i in range(len(self.atomic_numbers))]
        self.charge = int(charge)
        self.multiplicity = int(multiplicity)

    def exclusive_bond_score(self, start_xyz: np.ndarray, slot: int) -> float:
        """Same semantics as ``r3b.ReactionBundle.exclusive_bond_score``."""
        def bonds_of(ep_idx: int) -> Dict[Tuple[int, int], float]:
            return {(int(b["i"]), int(b["j"])): float(b["bond_order"])
                    for b in self.task_doc["endpoints"][ep_idx]["bonds"]}
        me, other = bonds_of(slot), bonds_of(1 - slot)
        mine = {p for p, o in me.items()
                if p not in other or other[p] != o}
        theirs = {p for p, o in other.items()
                  if p not in me or me[p] != o}
        tests = []
        for (i, j) in mine:
            tests.append(np.linalg.norm(start_xyz[i] - start_xyz[j]) < 1.9)
        for (i, j) in theirs:
            tests.append(np.linalg.norm(start_xyz[i] - start_xyz[j]) > 2.1)
        if not tests:
            return float("nan")
        return float(sum(tests)) / len(tests)


def _synth_wp2_task(rid: str, p1_doc: dict, charge: int,
                    multiplicity: int) -> dict:
    """Synthetic wp2-style task document from the P1 mapping alone.

    Endpoints are the P1 ``graph.r_bonds`` / ``graph.p_bonds`` expressed in
    endpoint-local indexing with the identity atom-map order (``m0001..``);
    :func:`build_key_atom_task` converts to the IRC row space via
    ``map_to_atoms`` and assigns endpoint sides by bond-set matching
    (endpoint[0] = R by construction of this synthesis).
    """
    map_entries = p1_doc.get("mapping", {}).get("map_to_atoms") or []
    n = len(map_entries)
    if not n:
        raise BuildError("p1_no_mapping", rid)

    def local(bonds: Sequence[Sequence[Any]]) -> List[dict]:
        out = []
        for b in bonds:
            out.append({"i": int(b[0]) - 1, "j": int(b[1]) - 1,
                        "bond_order": float(b[2])})
        return out

    graph = p1_doc.get("graph") or {}
    r_bonds = local(graph.get("r_bonds") or [])
    p_bonds = local(graph.get("p_bonds") or [])
    if not r_bonds or not p_bonds:
        raise BuildError("p1_endpoint_bonds_missing", rid)
    atom_map_ids = [f"m{i + 1:04d}" for i in range(n)]
    return {
        "reaction_id": rid,
        "charge": int(charge),
        "multiplicity": int(multiplicity),
        "endpoints": [
            {"atom_map_ids": atom_map_ids, "bonds": r_bonds},
            {"atom_map_ids": atom_map_ids, "bonds": p_bonds},
        ],
    }


def _split_segments(coords: np.ndarray) -> Tuple[int, np.ndarray, np.ndarray]:
    """Locate the inter-segment break; return (brk, seg1, seg2)."""
    n = int(coords.shape[0])
    if n < 2 * MIN_SEGMENT_FRAMES:
        raise BuildError("too_few_frames", f"n_frames={n}")
    steps = np.linalg.norm(np.diff(coords, axis=0), axis=2).mean(axis=1)
    brk = int(np.argmax(steps))
    med = float(np.median(steps))
    if not (0 < brk < n - 2):
        raise BuildError("segment_break_at_edge", f"brk={brk} n={n}")
    if not (steps[brk] > 8.0 * max(med, 1e-8)):
        raise BuildError("segment_break_not_resolved",
                         f"max step {steps[brk]:.4f} vs median {med:.4f}")
    seg1 = coords[:brk + 1]      # TS .. end1
    seg2 = coords[brk + 1:]      # TS' .. end2
    if seg1.shape[0] < MIN_SEGMENT_FRAMES or seg2.shape[0] < MIN_SEGMENT_FRAMES:
        raise BuildError("segment_too_short",
                         f"{seg1.shape[0]}/{seg2.shape[0]} frames")
    return brk, seg1, seg2


def _edit_direction_score(edits: Sequence[Tuple[int, int, float]],
                          end_a: np.ndarray, end_b: np.ndarray) -> float:
    """Comparative orientation evidence: fraction of edits on which ``end_a``
    behaves like the R side relative to ``end_b`` (forming bonds longer,
    breaking bonds shorter at the R end).  Symmetric: score(B) ~= 1-score(A).
    """
    votes = 0.0
    total = 0
    for (i, j, d) in edits:
        da = float(np.linalg.norm(end_a[i] - end_a[j]))
        db = float(np.linalg.norm(end_b[i] - end_b[j]))
        if abs(da - db) < 1e-6:
            continue
        total += 1
        if d > 0:      # forming: R end has the LONGER bond
            votes += 1.0 if da > db else 0.0
        else:          # breaking: R end has the SHORTER (still bonded) bond
            votes += 1.0 if da < db else 0.0
    if total == 0:
        return float("nan")
    return votes / total


def _decide_orientation(bundle: H5Bundle, end1: np.ndarray, end2: np.ndarray,
                        edits: Sequence[Tuple[int, int, float]],
                        g1_orientation: str) -> Tuple[str, dict]:
    """Which segment end is the R side?  Returns (which, evidence).

    Primary evidence is the comparative edit-direction score (all signed
    bond edits, both segment ends; validated 2026-10-22 against the G1
    orientation on a seeded 200-reaction audit: 198 agree / 0 disagree /
    2 ambiguous).  Exclusive-bond scores are recorded as secondary evidence.
    A hard disagreement with the G1 orientation excludes the reaction
    (data-corruption signal), as does an undecidable comparative score.
    """
    s1_r = bundle.exclusive_bond_score(end1, 0)
    s1_p = bundle.exclusive_bond_score(end1, 1)
    s2_r = bundle.exclusive_bond_score(end2, 0)
    s2_p = bundle.exclusive_bond_score(end2, 1)
    score = _edit_direction_score(edits, end1, end2)
    evidence: Dict[str, Any] = {
        "method": "edit_direction_comparative",
        "score_end1_r_like": (None if math.isnan(score) else score),
        "exclusive": {"end1_r": s1_r, "end1_p": s1_p,
                      "end2_r": s2_r, "end2_p": s2_p},
        "g1_orientation": g1_orientation,
    }
    if math.isnan(score):
        raise BuildError("orientation_undecidable", "no usable edits")
    if score >= 1.0 - ORIENTATION_COMPARATIVE_MARGIN:
        which = "R_first"
    elif score <= ORIENTATION_COMPARATIVE_MARGIN:
        which = "P_first"
    else:
        raise BuildError("orientation_ambiguous",
                         f"comparative score {score:.3f}")
    g1 = (g1_orientation or "").strip()
    if g1 in ("P_first", "R_first"):
        evidence["g1_disagreement"] = bool(g1 != which)
        if evidence["g1_disagreement"]:
            raise BuildError(
                "orientation_g1_conflict",
                f"frame evidence {which} vs G1 {g1} "
                f"(score {score:.3f})")
    else:
        evidence["g1_disagreement"] = False
    return which, evidence


def _process_curve(task: Dict[str, Any]) -> Dict[str, Any]:
    """Build one canonical curve + all six-table rows (worker-side, no I/O
    outside the staging npz and the P1 mapping — no truth access here)."""
    rid = task["rid"]
    coords = task["coordinates"]
    energies = task["energies"]
    z_h5 = task["atomic_numbers"]
    p1_doc = task["p1_doc"]
    ts_info = task["ts_info"]          # {charge, multiplicity, atomic_numbers}
    fold = int(task["fold"])
    g1_orientation = str(task.get("g1_orientation") or "")

    n = int(coords.shape[0])
    if energies is None or int(energies.shape[0]) != n:
        raise BuildError("energies_missing", f"n_frames={n}")
    z_p1 = [int(z) for z in ts_info["atomic_numbers"]]
    if z_h5 is not None and list(int(z) for z in z_h5) != z_p1:
        raise BuildError("z_mismatch_h5_ts", f"{z_h5[:6]} vs {z_p1[:6]}")

    brk, seg1, seg2 = _split_segments(coords)
    en1 = energies[:brk + 1]
    en2 = energies[brk + 1:]
    # TS-energy sanity: frame 0 must be (within tol of) the trajectory max
    e_ts = float(energies[0])
    e_ts2 = float(energies[brk + 1])
    e_max = float(np.max(energies))
    if e_ts < e_max - TS_ENERGY_MAX_TOL_EH and e_ts2 < e_max - TS_ENERGY_MAX_TOL_EH:
        raise BuildError("ts_energy_not_max",
                         f"E[0]={e_ts:.4f} E[brk+1]={e_ts2:.4f} "
                         f"max={e_max:.4f}")

    task_doc = _synth_wp2_task(rid, p1_doc, ts_info["charge"],
                               ts_info["multiplicity"])

    # provisional edits for orientation (row space, from P1 r/p bonds)
    row_of_map = {int(e["map"]): int(e["ts_irc_index"])
                  for e in p1_doc["mapping"]["map_to_atoms"]}
    b_r = {(min(row_of_map[int(b[0])], row_of_map[int(b[1])]),
            max(row_of_map[int(b[0])], row_of_map[int(b[1])])): float(b[2])
           for b in p1_doc["graph"]["r_bonds"]}
    b_p = {(min(row_of_map[int(b[0])], row_of_map[int(b[1])]),
            max(row_of_map[int(b[0])], row_of_map[int(b[1])])): float(b[2])
           for b in p1_doc["graph"]["p_bonds"]}
    edits = []
    for pair in sorted(set(b_r) | set(b_p)):
        d = float(b_p.get(pair, 0.0) - b_r.get(pair, 0.0))
        if abs(d) > 1e-9:
            edits.append((pair[0], pair[1], d))
    if not edits:
        raise BuildError("no_bond_edits", rid)

    bundle_probe = H5Bundle(rid, task_doc, p1_doc, task["p1_path"],
                            {"brA": seg1[::-1], "brB": seg2[::-1]},
                            {"brA": "A", "brB": "B"},
                            z_p1, ts_info["charge"], ts_info["multiplicity"])
    which, orient_evidence = _decide_orientation(
        bundle_probe, seg1[-1], seg2[-1], edits, g1_orientation)

    # canonical branches: endpoint -> TS, shared canonical TS = frame 0
    ts0 = coords[0]
    if which == "R_first":
        seg_r, en_r = seg1, en1
        seg_p, en_p = seg2, en2
    else:
        seg_r, en_r = seg2, en2
        seg_p, en_p = seg1, en1
    x_r = seg_r[::-1].copy()             # R endpoint .. TS
    x_p = seg_p[::-1].copy()             # P endpoint .. TS'
    en_x_r = en_r[::-1]
    en_x_p = en_p[::-1]
    x_r[-1] = ts0                        # one shared canonical TS frame
    x_p[-1] = ts0
    en_x_r[-1] = e_ts
    en_x_p[-1] = e_ts

    bundle = H5Bundle(rid, task_doc, p1_doc, task["p1_path"],
                      {"brA": x_r, "brB": x_p}, {"brA": "A", "brB": "B"},
                      z_p1, ts_info["charge"], ts_info["multiplicity"])
    try:
        curve = r3b.Curve(bundle)
    except (ValueError, KeyError) as exc:
        raise BuildError("curve_construction", str(exc)) from exc

    # canonical energies (node-aligned with curve.xyz)
    en_curve = np.concatenate([en_x_r, en_x_p[-2::-1]])
    # gauge gate on the canonical curve (frozen native thresholds)
    continuity = verify_frame_continuity(curve.xyz)
    thr = continuity.get("thresholds", {})
    ev = continuity.get("evidence", {})
    consec_ok = (
        float(ev.get("max_consecutive_rotation_deg", math.inf))
        <= float(thr.get("max_consecutive_rotation_deg", 2.0))
        and float(ev.get("max_consecutive_translation_angstrom", math.inf))
        <= float(thr.get("max_consecutive_translation_angstrom", 0.1)))
    gauge_excluded = not consec_ok

    labels = {"forward": r3b.solve_curve_direction(curve, "forward"),
              "reverse": r3b.solve_curve_direction(curve, "reverse")}
    flags, cstats = r3b.conflict_audit_curve(curve, labels)
    overflow_result = r3b.overflow_check(bundle, curve)

    rows: Dict[str, List[str]] = {name: [] for name in TABLES_ORDER}
    out = {
        "rows": rows,
        "ledger": {},
        "edge_flat_len": 0,
        "excluded": None,
    }

    intents = []
    for direction in ("forward", "reverse"):
        if direction == "forward":
            g_src, g_tgt = curve.g_r, curve.g_p
            sign = 1.0
        else:
            g_src, g_tgt = curve.g_p, curve.g_r
            sign = -1.0
        intent = {
            "intent_id": "{}_{}".format(
                curve.curve_id, "fwd" if direction == "forward" else "rev"),
            "curve_id": curve.curve_id,
            "direction": direction,
            "reverse_of_intent_id": (
                f"{curve.curve_id}_rev" if direction == "forward"
                else f"{curve.curve_id}_fwd"),
            "intent_pair_id": f"pair_{curve.curve_id}",
            "intent_underdetermined": (not curve.edits),
            "g_src": g_src,
            "g_tgt": g_tgt,
            "edits": [[i, j, sign * d] for i, j, d in curve.edits],
            "cv_spec": curve.cv_spec,
            "intent_role": (curve.intent_role if direction == "forward"
                            else {
                                "forming": curve.intent_role["breaking"],
                                "breaking": curve.intent_role["forming"],
                                "both": curve.intent_role["both"]}),
            "rules": {
                "relabel_invariant": True,
                "reverse_rule": "swap G_src/G_tgt, negate delta_B_intent",
                "d0_angstrom": MAIN_D0,
                "teacher": TEACHER_VERSION,
                "forbidden_forward_inputs": [
                    "future_frames", "ts_coords", "true_progress",
                    "branch_id", "split_or_family", "energy_or_gradient"],
            },
            "intent_version": INTENT_VERSION,
        }
        intents.append(intent)
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow([
            intent["intent_id"], intent["curve_id"], direction,
            intent["reverse_of_intent_id"], intent["intent_pair_id"],
            r3b._fmt_bool(intent["intent_underdetermined"]),
            r3b.jdump(g_src), r3b.jdump(g_tgt),
            r3b.jdump(intent["edits"]),
            r3b.jdump(curve.cv_spec) if curve.cv_spec else "null",
            r3b.jdump(intent["intent_role"]),
            r3b.jdump(intent["rules"]),
            INTENT_VERSION,
        ])
        rows["reaction_intent.csv"].append(buf.getvalue())

    n_nodes = curve.n
    n_rows_curve = 2 * n_nodes
    curve_weight = r3b.CURVE_WEIGHT_TARGET / n_rows_curve
    status_counts: Counter = Counter()
    dispositions: Counter = Counter()

    source_path = "h5:{}".format(rid)
    for node in range(n_nodes):
        xyz = curve.xyz[node]
        for intent in intents:
            direction = intent["direction"]
            sol = labels[direction][node]
            if gauge_excluded:
                status, mask, disp, gauge = (
                    "masked", True, "gauge_unresolved_excluded",
                    "unresolved_masked")
            elif sol["terminal"]:
                status, mask, disp, gauge = (
                    "terminal", True, "terminal_no_target", "local_frame")
            elif not sol["eligible"]:
                status, mask, disp, gauge = (
                    "masked", True, "no_future_step", "local_frame")
            else:
                status, mask, disp, gauge = (
                    "ok", False, "train_eligible", "local_frame")
            if status == "ok":
                cflag = flags.get((node, direction))
                if cflag == "conflict":
                    status, mask = "conflict", True
                    disp = "conflict_near_input_masked"
                elif cflag == "near_conflict":
                    disp = "near_conflict_downweighted"
            label_weight = (1.0 if status == "ok" else
                            r3b.NEAR_CONFLICT_WEIGHT
                            if disp == "near_conflict_downweighted"
                            else 0.0)
            # overflow_check is DIAGNOSTIC-ONLY in this lineage (recorded in
            # curve_quality.note + build ledger): the raw IRC terminal frames
            # are not fully relaxed endpoints, so a strict edit-direction
            # gate would quarantine marginal physics rather than mapping
            # errors (the P1 status filter already removed
            # ts_irc_endpoint_mismatch / source_structure_mismatch).

            if mask and status not in ("terminal", "conflict"):
                u_out = None
                t_node = None
                t_alpha = None
            elif status == "terminal":
                u_out = None
                t_node = None
                t_alpha = None
            else:
                u_out = sol["u_raw"]
                t_node = sol["target_node_index"]
                t_alpha = sol["target_alpha"]

            sample_weight = label_weight * curve_weight
            feat = r3b.feature_payload(bundle, curve, node, intent, MAIN_D0)
            sol_row = {
                "label_status": status,
                "mask": mask,
                "target_node_index": t_node,
                "target_alpha": t_alpha,
                "u_raw": u_out,
            }
            feature_sha = r3b.sha256_json(feat)
            label_sha = r3b.sha256_json(r3b.label_payload(sol_row, MAIN_D0))
            sample_id = (f"{curve.curve_id}:{node}:"
                         f"{intent['intent_id']}:{MAIN_D0:g}")
            cx, cy, cz = (float(v) for v in xyz.mean(axis=0))
            status_counts[status] += 1
            dispositions[disp] += 1
            buf = io.StringIO()
            w = csv.writer(buf, lineterminator="\n")
            w.writerow([
                sample_id, curve.curve_id, node, intent["intent_id"],
                repr(MAIN_D0),
                repr(cx), repr(cy), repr(cz),
                repr(label_weight), repr(curve_weight),
                repr(sample_weight), status, feature_sha, label_sha,
                "false", gauge,
                t_node if t_node is not None else "",
                r3b._fmt_float(t_alpha),
                (json.dumps([float(v) for v in u_out.ravel()])
                 if u_out is not None else ""),
                "",
                "",
                r3b._fmt_bool(mask),
                fold,
                disp,
                json.dumps([float(v) for v in xyz.ravel()]),
                f"[{curve.n_atoms},3]",
            ])
            rows["sample_view.csv"].append(buf.getvalue())

    # edges (PD-9; chunk-local offsets fixed at assembly)
    fwd_id = f"{curve.curve_id}_fwd"
    rev_id = f"{curve.curve_id}_rev"
    atom_map_sha = r3b.sha256_json({
        "atom_map_ids": bundle.atom_map_ids,
        "rule": "row i <-> map i+1 <-> IRC atom row i (g1 invariant)",
    })
    chunk_flat = 0
    for intent_id, direction in ((fwd_id, "forward"), (rev_id, "reverse")):
        for k in range(n_nodes - 1):
            if direction == "forward":
                src, dst = k, k + 1
                delta = curve.xyz[k + 1] - curve.xyz[k]
            else:
                src, dst = k + 1, k
                delta = -(curve.xyz[k + 1] - curve.xyz[k])
            flat = [float(v) for v in delta.ravel()]
            buf = io.StringIO()
            w = csv.writer(buf, lineterminator="\n")
            w.writerow([
                curve.curve_id, intent_id, k, src, dst,
                json.dumps(flat), f"[{curve.n_atoms},3]",
                json.dumps([chunk_flat, chunk_flat + len(flat)]),
                atom_map_sha, r3b.sha256_flat_f64(flat), CANONICAL_UNITS,
                "true",
            ])
            rows["irc_edge.csv"].append(buf.getvalue())
            chunk_flat += len(flat)
    out["edge_flat_len"] = chunk_flat

    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([
        curve.curve_id, f"irc_{curve.curve_id}", n_nodes,
        2 * (n_nodes - 1), curve.ts_index, 0, n_nodes - 1,
        source_path,
        repr(float(en_curve[0])), repr(float(en_curve[-1])),
    ])
    rows["irc_curve.csv"].append(buf.getvalue())

    z_list = list(bundle.atomic_numbers)
    for node in range(n_nodes):
        x, y, z = (float(v) for v in curve.xyz[node].mean(axis=0))
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow([
            curve.curve_id, node, repr(x), repr(y), repr(z),
            repr(float(curve.progress[node])),
            1 if node == curve.ts_index else 0,
            1 if node in (0, n_nodes - 1) else 0,
            repr(float(en_curve[node])), "", "ok", r3b.jdump(z_list),
            json.dumps([float(v) for v in curve.xyz[node].ravel()]),
            f"[{curve.n_atoms},3]",
        ])
        rows["irc_node.csv"].append(buf.getvalue())

    n_ok = int(status_counts.get("ok", 0))
    quality_status = (
        "gauge_unresolved_excluded" if gauge_excluded else
        "ok")
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow([
        curve.curve_id, quality_status,
        repr(n_ok / n_rows_curve),
        r3b.jdump({"n_ok": n_ok, "n_rows": n_rows_curve}),
        "null",
        r3b.jdump({
            "p1_mapping_sha256": sha256_file(Path(task["p1_path"])),
            "ts_join_maxabs": curve.ts_join_maxabs,
            "frame_continuity": continuity,
            "binding_evidence": curve._binding_evidence,
            "endpoint_sides": {"endpoint_0": list(curve.sides)[0],
                               "endpoint_1": list(curve.sides)[1]},
            "orientation": orient_evidence,
            "overflow_check": overflow_result,
        }),
    ])
    rows["curve_quality.csv"].append(buf.getvalue())

    # branch diagnostics (one row per segment)
    for bid, label, seg, side in (
            ("brA", "A", x_r, "R"), ("brB", "B", x_p, "P")):
        cont_b = verify_frame_continuity(seg)
        buf = io.StringIO()
        w = csv.writer(buf, lineterminator="\n")
        w.writerow([
            curve.curve_id, bid, label, side, int(seg.shape[0]),
            "", "", "", "",
            repr(float(cont_b["evidence"]["max_consecutive_rotation_deg"])),
            repr(float(cont_b["evidence"]
                       ["max_consecutive_translation_angstrom"])),
            repr(float(np.linalg.norm(np.diff(seg, axis=0), axis=2).max())),
            source_path,
        ])
        rows["branch_diagnostics.csv"].append(buf.getvalue())

    out["ledger"] = {
        "n_nodes": n_nodes,
        "ts_index": curve.ts_index,
        "fold": fold,
        "key_rows": list(curve.key_rows),
        "n_edits": len(curve.edits),
        "segment_break": brk,
        "orientation": orient_evidence,
        "quality_status": quality_status,
        "overflow_consistent": bool(overflow_result["consistent"]),
        "conflict_audit": cstats,
        "frame_continuity_evidence": continuity.get("evidence"),
        "ts_join_maxabs": curve.ts_join_maxabs,
        "ts_energy": e_ts,
        "ts_prime_energy": e_ts2,
        "endpoint_energies": [float(en_curve[0]), float(en_curve[-1])],
        "status_counts": dict(status_counts),
        "dispositions": dict(dispositions),
        "p1_mapping_sha256": sha256_file(Path(task["p1_path"])),
        "frames_sha256": task["frames_sha256"],
        "energies_sha256": task["energies_sha256"],
    }
    return out


def _worker_chunk(job: Dict[str, Any]) -> Dict[str, Any]:
    """Process one chunk of reactions; write per-table CSV fragments."""
    staging = Path(job["staging_path"])
    frag_dir = Path(job["frag_dir"])
    frag_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(staging)
    per_table: Dict[str, List[str]] = {name: [] for name in TABLES_ORDER}
    ledger: Dict[str, Any] = {}
    exclusions: Dict[str, Any] = {}
    edge_flat_total = 0
    for item in job["items"]:
        rid = item["rid"]
        try:
            coords = np.asarray(
                data[f"{rid}::coordinates"], dtype=np.float64)
            en_key = f"{rid}::energies"
            energies = (np.asarray(data[en_key], dtype=np.float64)
                        if en_key in data else None)
            z_key = f"{rid}::atomic_numbers"
            z_h5 = (tuple(int(v) for v in data[z_key])
                    if z_key in data else None)
            p1_path = Path(job["p1_root"]) / item["p1_relpath"]
            p1_doc = json.loads(p1_path.read_text(encoding="utf-8"))
            result = _process_curve({
                "rid": rid,
                "coordinates": coords,
                "energies": energies,
                "atomic_numbers": z_h5,
                "p1_doc": p1_doc,
                "p1_path": str(p1_path),
                "ts_info": item["ts_info"],
                "fold": item["fold"],
                "g1_orientation": item.get("g1_orientation") or "",
                "frames_sha256": item["frames_sha256"],
                "energies_sha256": item["energies_sha256"],
            })
        except BuildError as exc:
            exclusions[rid] = {"reason": exc.reason, "detail": exc.detail}
            continue
        except Exception as exc:  # noqa: BLE001 - recorded, never silent
            exclusions[rid] = {"reason": "build_exception",
                               "detail": f"{type(exc).__name__}: {exc}"}
            continue
        for name in TABLES_ORDER:
            per_table[name].extend(result["rows"][name])
        ledger[rid] = result["ledger"]
        edge_flat_total += int(result["edge_flat_len"])
    paths = {}
    for name in TABLES_ORDER:
        frag = frag_dir / name
        frag.write_text("".join(per_table[name]), encoding="utf-8")
        paths[name] = str(frag)
    # persist per-curve evidence (ledger + exclusions) inside the fragment dir
    (frag_dir / "ledger.jsonl").write_text(
        "".join(json.dumps({"curve_id": rid, **entry}, sort_keys=True,
                           ensure_ascii=False, default=str) + "\n"
                for rid, entry in sorted(ledger.items())),
        encoding="utf-8")
    (frag_dir / "exclusions.jsonl").write_text(
        "".join(json.dumps({"curve_id": rid, **entry}, sort_keys=True,
                           ensure_ascii=False, default=str) + "\n"
                for rid, entry in sorted(exclusions.items())),
        encoding="utf-8")
    return {
        "chunk_index": job["chunk_index"],
        "frag_paths": paths,
        "ledger": ledger,
        "exclusions": exclusions,
        "edge_flat_total": edge_flat_total,
        "n_built": len(ledger),
    }


# ---------------------------------------------------------------------------
# stage: audited streaming extraction -> staging chunks
# ---------------------------------------------------------------------------


def _p1_relpath(rid: str) -> str:
    number = int(rid.split("_")[-1])
    return f"{number // 1000:05d}/{rid}.json"


def cmd_stage(args: argparse.Namespace, repo: Path) -> dict:
    cohort = json.loads(Path(args.cohort).read_text(encoding="utf-8"))
    pool = [str(c) for c in cohort["pool_candidates"]]
    dev = [str(c) for c in cohort["dev"]]
    build_order = pool + dev                      # pool first (rank), then dev
    if args.limit is not None:
        build_order = build_order[: args.limit]
    wanted = set(build_order)
    fold_of = {rid: i % 3 for i, rid in enumerate(build_order)}

    # scope assertion BEFORE the archive is opened (bridge re-checks)
    split_doc = json.loads((repo / G3P1_SPLIT).read_text(encoding="utf-8"))
    scope = set()
    for subset in ("train_fit", "train_dev", "train_calibration"):
        scope.update(str(v) for v in split_doc["subsets"][subset])
    outside = sorted(wanted - scope)
    if outside:
        raise SystemExit(
            f"SCOPE_VIOLATION: {len(outside)} requested reactions are outside "
            f"the frozen train-side subsets (e.g. {outside[:3]})")

    # bulk audited TS table read (charge/multiplicity/atomic_numbers only)
    ts_table = truth_reader.load_ts_table(
        allow_truth=True, manifests_dir=repo / MANIFESTS_DIR)
    ts_info: Dict[str, dict] = {}
    for row in ts_table.to_pylist():
        rid = str(row["reaction_id"])
        if rid in wanted:
            ts_info[rid] = {
                "charge": int(row["charge"]),
                "multiplicity": int(row["multiplicity"]),
                "atomic_numbers": [int(z) for z in row["atomic_numbers"]],
            }
    missing_ts = sorted(wanted - set(ts_info))
    if missing_ts:
        raise SystemExit(
            f"TS table misses {len(missing_ts)} wanted reactions "
            f"(e.g. {missing_ts[:3]})")

    out_dir = Path(args.out)
    staging_dir = out_dir / "staging_chunks"
    staging_dir.mkdir(parents=True, exist_ok=True)
    log_path = repo / MANIFESTS_DIR / "truth_access_log.jsonl"
    log_lines_before = 0
    if log_path.is_file():
        log_lines_before = sum(
            1 for ln in log_path.read_text(encoding="utf-8").splitlines()
            if ln.strip())

    n_chunks = (len(build_order) + CHUNK_SIZE - 1) // CHUNK_SIZE
    chunk_of = {rid: i // CHUNK_SIZE for i, rid in enumerate(build_order)}
    buffers: Dict[int, Dict[str, dict]] = {}
    remaining = {i: int(np.sum([chunk_of[r] == i for r in build_order]))
                 for i in range(n_chunks)}
    extraction: Dict[str, dict] = {}
    t0 = time.time()
    seen = 0
    for frames in truth_source.stream_scoped_irc_frames(
        frozenset(wanted),
        allow_truth=True,
        manifests_dir=repo / MANIFESTS_DIR,
        split_dir=repo / G3P1_SPLIT.rsplit("/", 1)[0],
    ):
        rid = str(frames["reaction_id"])
        if rid not in wanted:
            continue
        seen += 1
        coords = np.asarray(frames["coordinates"], dtype=np.float32)
        energies = (None if frames.get("EHG") is None
                    else np.asarray(frames["EHG"], dtype=np.float64))
        z = frames.get("atomic_numbers")
        extraction[rid] = {
            "n_frames": int(frames["n_frames"]),
            "n_atoms": int(frames["n_atoms"]),
            "has_forces": bool(frames["has_forces"]),
            "coordinates_sha256": sha256_bytes(coords.tobytes()),
            "energies_sha256": (sha256_bytes(energies.tobytes())
                                if energies is not None else None),
            "atomic_numbers_sha256": (sha256_bytes(
                np.asarray(z, dtype=np.int64).tobytes()) if z else None),
            "bytes": int(coords.nbytes
                         + (energies.nbytes if energies is not None else 0)),
        }
        ci = chunk_of[rid]
        buffers.setdefault(ci, {})[rid] = {
            "coordinates": coords,
            "energies": energies,
            "atomic_numbers": (np.asarray(z, dtype=np.int64) if z else None),
        }
        if len(buffers[ci]) == remaining[ci]:
            np.savez(staging_dir / f"chunk_{ci:04d}.npz",
                     **{f"{rid}::{k}": v[k]
                        for rid, v in buffers[ci].items()
                        for k in ("coordinates", "energies", "atomic_numbers")
                        if v[k] is not None})
            del buffers[ci]
        if seen % 1000 == 0:
            print(f"[stage] {seen}/{len(wanted)} reactions "
                  f"({time.time() - t0:.0f}s)", flush=True)

    missing = sorted(wanted - set(extraction))
    log_lines_after = sum(
        1 for ln in log_path.read_text(encoding="utf-8").splitlines()
        if ln.strip())
    log_sha = sha256_file(log_path)
    # copy this build's slice of the shared audit log
    log_slice = log_path.read_text(encoding="utf-8").splitlines()[
        log_lines_before:log_lines_after]
    slice_path = out_dir / "truth_access_log.jsonl"
    slice_path.write_text("\n".join(log_slice) + ("\n" if log_slice else ""),
                          encoding="utf-8")

    manifest = {
        "schema": "v53mp_extraction_manifest_v1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "tool_version": TOOL_VERSION,
        "cohort": {"path": args.cohort, "sha256": sha256_file(Path(args.cohort))},
        "build_order": build_order,
        "chunk_size": CHUNK_SIZE,
        "n_chunks": n_chunks,
        "fold_rule": "fold = build_position % 3 (pool and dev curves; "
                     "confirm-listed reactions are never built)",
        "wanted": len(wanted),
        "staged": len(extraction),
        "missing_from_archive": missing,
        "per_reaction": extraction,
        "truth_access": {
            "bridge": "pes2ts_core.pesn.offline.data.truth_source."
                      "stream_scoped_irc_frames",
            "ts_table": TS_PARQUET_VIA,
            "shared_log": {
                "path": str(log_path.relative_to(repo)),
                "lines_before": log_lines_before,
                "lines_after": log_lines_after,
                "sha256": log_sha,
            },
            "build_log_copy": "truth_access_log.jsonl",
            "hard_gate": "no UNAUTHORIZED access (every read recorded by "
                         "the audited accessor; scope enforced before the "
                         "archive is opened)",
        },
    }
    (out_dir / "extraction_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return {
        "wanted": len(wanted), "staged": len(extraction),
        "missing": len(missing),
        "wall_clock_s": round(time.time() - t0, 1),
    }


# ---------------------------------------------------------------------------
# build: parallel per-chunk curve construction
# ---------------------------------------------------------------------------


def cmd_build(args: argparse.Namespace, repo: Path) -> dict:
    out_dir = Path(args.out)
    staging_dir = out_dir / "staging_chunks"
    frag_root = out_dir / "fragments"
    manifest = json.loads(
        (out_dir / "extraction_manifest.json").read_text(encoding="utf-8"))
    build_order = manifest["build_order"]
    if args.limit is not None:
        build_order = build_order[: args.limit]
    fold_of = {rid: i % 3 for i, rid in enumerate(build_order)}

    ts_table = None
    ts_info: Dict[str, dict] = {}
    ts_path = repo / "data/ground_truth/ts.parquet"
    # NOTE: reads the TS parquet via the audited bulk accessor only when it
    # was not staged; staging keeps charge/multiplicity out of the npz, so
    # the build re-reads the parquet through the SAME audited accessor
    # (one audit line) — no per-reaction truth reads happen in workers.
    ts_table = truth_reader.load_ts_table(
        allow_truth=True, manifests_dir=repo / MANIFESTS_DIR)
    wanted = set(build_order)
    for row in ts_table.to_pylist():
        rid = str(row["reaction_id"])
        if rid in wanted:
            ts_info[rid] = {
                "charge": int(row["charge"]),
                "multiplicity": int(row["multiplicity"]),
                "atomic_numbers": [int(z) for z in row["atomic_numbers"]],
            }

    extraction = manifest["per_reaction"]
    # G1 orientation cross-check evidence (recorded per curve in the ledger)
    orient_of: Dict[str, str] = {}
    s0a_parquet = repo / "outputs/pesn/v53mp/s0a/eligible_reactions.parquet"
    if s0a_parquet.is_file():
        import pyarrow.parquet as pq

        for row in pq.read_table(
                s0a_parquet, columns=["reaction_id", "p1_orientation"]
        ).to_pylist():
            orient_of[str(row["reaction_id"])] = str(
                row["p1_orientation"] or "")
    n_chunks = (len(build_order) + CHUNK_SIZE - 1) // CHUNK_SIZE
    jobs = []
    for ci in range(n_chunks):
        chunk_ids = build_order[ci * CHUNK_SIZE:(ci + 1) * CHUNK_SIZE]
        frag_dir = frag_root / f"chunk_{ci:04d}"
        marker = frag_dir / "_done.json"
        if marker.is_file() and not args.force:
            continue
        items = []
        for rid in chunk_ids:
            ext = extraction.get(rid)
            if ext is None:
                continue
            items.append({
                "rid": rid,
                "p1_relpath": _p1_relpath(rid),
                "ts_info": ts_info[rid],
                "fold": fold_of[rid],
                "g1_orientation": orient_of.get(rid, ""),
                "frames_sha256": ext["coordinates_sha256"],
                "energies_sha256": ext["energies_sha256"],
            })
        jobs.append({
            "chunk_index": ci,
            "staging_path": str(staging_dir / f"chunk_{ci:04d}.npz"),
            "frag_dir": str(frag_dir),
            "p1_root": str(repo / P1_REL),
            "items": items,
        })

    t0 = time.time()
    results: List[dict] = []
    workers = max(1, int(args.workers))
    if workers == 1:
        for job in jobs:
            results.append(_worker_chunk(job))
            print(f"[build] chunk {job['chunk_index']} done "
                  f"({time.time() - t0:.0f}s)", flush=True)
    else:
        ctx = get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            for result in pool.imap_unordered(_worker_chunk, jobs):
                results.append(result)
                print(f"[build] chunk {result['chunk_index']} done "
                      f"({result['n_built']} built, "
                      f"{len(result['exclusions'])} excluded, "
                      f"{time.time() - t0:.0f}s)", flush=True)
    for result in results:
        marker = frag_root / f"chunk_{result['chunk_index']:04d}" / "_done.json"
        marker.write_text(json.dumps({
            "chunk_index": result["chunk_index"],
            "n_built": result["n_built"],
            "n_excluded": len(result["exclusions"]),
            "edge_flat_total": result["edge_flat_total"],
        }, sort_keys=True) + "\n", encoding="utf-8")
    return {"chunks_done": len(results),
            "wall_clock_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# assemble: concat fragments -> six tables + manifests + ledger
# ---------------------------------------------------------------------------


def _concat_simple(frag_paths: List[str], target: Path, header: List[str]) -> None:
    with target.open("w", encoding="utf-8", newline="") as out_fh:
        out_fh.write(",".join(header) + "\n")
        for frag in frag_paths:
            with open(frag, "r", encoding="utf-8") as in_fh:
                shutil.copyfileobj(in_fh, out_fh, 1 << 20)


def _concat_edges(frag_paths: List[str], target: Path,
                  header: List[str]) -> None:
    """Concatenate irc_edge fragments with global PD-9 running offsets."""
    offset = 0
    with target.open("w", encoding="utf-8", newline="") as out_fh:
        w = csv.writer(out_fh, lineterminator="\n")
        w.writerow(header)
        for frag in frag_paths:
            with open(frag, "r", encoding="utf-8", newline="") as in_fh:
                reader = csv.reader(in_fh)
                for row in reader:
                    if not row:
                        continue
                    local = json.loads(row[7])
                    n_flat = int(local[1]) - int(local[0])
                    row[7] = json.dumps([offset, offset + n_flat])
                    offset += n_flat
                    w.writerow(row)


def cmd_assemble(args: argparse.Namespace, repo: Path) -> dict:
    out_dir = Path(args.out)
    frag_root = out_dir / "fragments"
    manifest_path = out_dir / "extraction_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    build_order = manifest["build_order"]
    if args.limit is not None:
        build_order = build_order[: args.limit]
    n_chunks = (len(build_order) + CHUNK_SIZE - 1) // CHUNK_SIZE

    frag_by_chunk: Dict[int, Dict[str, str]] = {}
    ledger: Dict[str, Any] = {}
    exclusions: Dict[str, Any] = {}
    for ci in range(n_chunks):
        frag_dir = frag_root / f"chunk_{ci:04d}"
        marker = frag_dir / "_done.json"
        if not marker.is_file():
            raise SystemExit(f"fragment chunk {ci} not built ({frag_dir})")
        frag_by_chunk[ci] = {name: str(frag_dir / name) for name in TABLES_ORDER}
        for line in (frag_dir / "ledger.jsonl").read_text(
                encoding="utf-8").splitlines():
            if line.strip():
                entry = json.loads(line)
                ledger[entry.pop("curve_id")] = entry
        excl_path = frag_dir / "exclusions.jsonl"
        if excl_path.is_file():
            for line in excl_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    exclusions[entry.pop("curve_id")] = entry

    t0 = time.time()
    for name in TABLES_ORDER:
        paths = [frag_by_chunk[ci][name] for ci in range(n_chunks)]
        target = out_dir / name
        if name == "irc_edge.csv":
            _concat_edges(paths, target, TABLE_HEADERS[name])
        else:
            _concat_simple(paths, target, TABLE_HEADERS[name])
        print(f"[assemble] {name} written ({time.time() - t0:.0f}s)",
              flush=True)

    # merge per-chunk ledgers from fragment sidecar json?  The chunk ledger
    # entries were returned to the driver at build time; rebuild them here
    # from the per-curve fragment row of curve_quality for the summary only.
    status_counts: Counter = Counter()
    dispositions: Counter = Counter()
    n_samples = 0
    quality_counts: Counter = Counter()
    with open(out_dir / "sample_view.csv", "r", encoding="utf-8",
              newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            n_samples += 1
            status_counts[row["label_status"]] += 1
            dispositions[row["disposition"]] += 1
    curve_ids: List[str] = []
    with open(out_dir / "irc_curve.csv", "r", encoding="utf-8",
              newline="") as fh:
        for row in csv.DictReader(fh):
            curve_ids.append(row["curve_id"])
    with open(out_dir / "curve_quality.csv", "r", encoding="utf-8",
              newline="") as fh:
        for row in csv.DictReader(fh):
            quality_counts[row["quality_status"]] += 1

    # truth access log: this build DID read the truth (audited); the slice
    # copy was written by `stage`.
    coverage = {
        "n_curves": len(curve_ids),
        "n_samples": n_samples,
        "samples_by_label_status": dict(status_counts),
        "samples_by_disposition": dict(dispositions),
        "curve_quality": dict(quality_counts),
    }

    data_files = list(TABLES_ORDER[:6]) + ["branch_diagnostics.csv"]
    dataset_sha = r3b.sha256_json(
        [[name, sha256_file(out_dir / name)] for name in data_files])
    data_manifest = {
        "version": "r3_data_manifest_v1",
        "schema_version": "r3_data_manifest_v1",
        "table_schema_version": SCHEMA_VERSION,
        "build_tool_version": TOOL_VERSION,
        "dataset_sha256": dataset_sha,
        "files": [{"path": name, "sha256": sha256_file(out_dir / name)}
                  for name in data_files],
    }
    (out_dir / "data_manifest.json").write_text(
        json.dumps(data_manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8")

    teacher_manifest = {
        "version": "r3_teacher_manifest_v1",
        "schema_version": "r3_teacher_manifest_v1",
        "build_tool_version": TOOL_VERSION,
        "teacher_version": TEACHER_VERSION,
        "intent_version": INTENT_VERSION,
        "d0_angstrom": MAIN_D0,
        "spec": {
            "solver": "pes2ts_core.pesn.offline.data.key_atom_teacher."
                      "solve_teacher (frozen, frame_layer=native_verified_v1)",
            "contact": "earliest root of RMS_K(Y(alpha)-X_t)=d0 along the "
                       "intent direction, piecewise-linear segments, "
                       "root tol 1e-8",
            "cap_angstrom": TEACHER_CAP_ANGSTROM,
            "terminal": "path exhausted below d0 -> label_status=terminal, "
                        "mask=true, no fabricated action",
            "no_future_step": "entire future identical to X_t -> masked",
            "reverse": "reverse intent re-reads the reversed path in the "
                       "same stored frame (label involution by construction)",
            "raw_vs_derived": "raw irc_edge.delta_xyz and derived "
                              "teacher_action_u are stored and hashed "
                              "separately",
            "frame_semantics": (
                "raw stored coordinates of the quarantined IRC archive "
                "(audited bridge reads; ts_first_dual_branch layout: frame 0 "
                "= TS, two direction segments, canonical curve "
                "[seg_R reversed] + [seg_P[1:]] with one shared TS); "
                "consecutive-frame continuity gated at the frozen native "
                "thresholds (2 deg / 0.1 A); failures masked "
                "(gauge_unresolved_excluded), never silently re-framed"),
        },
        "files": [
            {"path": name, "sha256": sha256_file(out_dir / name)}
            for name in ("sample_view.csv", "irc_edge.csv",
                         "reaction_intent.csv")
        ],
    }
    (out_dir / "teacher_manifest.json").write_text(
        json.dumps(teacher_manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8")

    ledger_doc = {
        "schema_version": "pes2ts_r3_v53mp_build_ledger_v1",
        "tool_version": TOOL_VERSION,
        "table_schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "teacher": teacher_manifest["spec"],
        "lineage": {
            "trajectory_source": {
                "kind": "quarantined B3LYP-D3/TZVP IRC archive via the "
                        "audited, scope-checked pesn truth bridge",
                "manifest": "data/manifests/truth_manifest.json",
                "cohort": manifest["cohort"],
            },
            "key_atoms_and_sides": {
                "path": P1_REL,
                "kind": "G1 P1 mapping bond_events (key atoms); endpoint "
                        "graphs synthesized from graph.r_bonds/p_bonds; "
                        "R/P side from frame-level bond evidence with G1 "
                        "orientation cross-check",
            },
            "ts_table": TS_PARQUET_VIA,
        },
        "coverage": coverage,
        "exclusions": {
            "total": len(exclusions),
            "by_reason": dict(Counter(
                e["reason"] for e in exclusions.values())),
            "per_curve": exclusions,
        },
        "curves": ledger,
        "truth_access": manifest["truth_access"],
        "curve_order": "build_order (pool candidates by cohort rank, then "
                       "dev candidates by cohort rank)",
        "notes": [
            "confirm-listed reactions are never read or built in this "
            "campaign stage",
            "energies stored in irc_node.energy / irc_curve.energy_a/b are "
            "B3LYP-D3/TZVP provenance values (never model inputs)",
        ],
    }
    (out_dir / "build_ledger_r3v53mp.json").write_text(
        json.dumps(ledger_doc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    return {"coverage": coverage, "dataset_sha256": dataset_sha,
            "wall_clock_s": round(time.time() - t0, 1)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="r3_v53mp_build_tables_h5")
    parser.add_argument("cmd", choices=("stage", "build", "assemble", "all"))
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--cohort",
                        default="outputs/pesn/v53mp/cohort/v53mp_cohort_v1.json")
    parser.add_argument("--out", default="outputs/pesn/v53mp/data")
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {}
    if args.cmd in ("stage", "all"):
        summary["stage"] = cmd_stage(args, repo)
    if args.cmd in ("build", "all"):
        summary["build"] = cmd_build(args, repo)
    if args.cmd in ("assemble", "all"):
        summary["assemble"] = cmd_assemble(args, repo)
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
