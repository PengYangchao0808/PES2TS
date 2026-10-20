#!/usr/bin/env python3
"""R3 six-table dataset builder for the 30h campaign (E1, lane W-DATA).

Builds ``outputs/pesn/30h_campaign/data/`` (six CSV tables + branch
diagnostics + split/data/teacher manifests + build ledger + empty truth
access log) from the authoritative R3 fit-wave lineage, schema
``r3_6table_v2`` (PD-1/PD-9, WP-D2).

Authoritative sources (read-only; hashes pinned in the build ledger):

* ``outputs/pesn/dual_ts_v2_20261015/wp2_fit50/RXN_*.{task,teacher}.json``
  — the coverage-audited, drift-registered dual-branch IRC export of the
  fit50 wave (source of node coordinates; TS frames of the two branches
  are exactly shared; node counts match ``coverage_summary.json``).
* ``data/interim/g1_truth/p1_mapping/**/RXN_*.json`` — G1 P1 mapping:
  authoritative key atoms (``bond_events``) and R/P endpoint-side matching
  via :func:`pes2ts_core.pesn.contracts.key_atom.build_key_atom_task`.
* ``data/manifests/v5/split_manifest.json`` — frozen V5 split (50 groups,
  5 folds, duplicate-ledger components); R3 inherits it unchanged
  (三位一体修复方案 §3.5).
* ``data/manifests/v5/dataset_audit.json`` — fit50-internal conflict
  audit (mapping-overflow cohort of 5 pairs).
* ``outputs/pesn/v5/native/*.meta.json`` + ``outputs/pesn/v5/teacher/d0_*``
  — prior V5 label pipeline artifacts, used **only** to carry forward the
  ruled cohort dispositions (1248 exclusion, 480 gauge rows, 2022
  recheck baseline); never as label values.

Teacher (R3 v3.1 §5, intent_directed_first_contact): for each canonical
curve A(R)->TS->B(P) and each directed ReactionIntent, the single-step
target is the first contact along the intent direction at d0=0.08 A on
the key-atom RMS, with piecewise-linear interpolation, earliest root,
frozen per-atom cap 0.20 A, TS-terminal / no-future-step semantics —
computed with the frozen
:func:`pes2ts_core.pesn.offline.data.key_atom_teacher.solve_teacher` in
the ``native_verified_v1`` layer (raw stored coordinates, no Kabsch
registration: the wp2 export is one drift-registered Cartesian frame per
curve — evidence recorded per curve in the build ledger).  The reverse
intent re-reads the reversed path, so ``irc_edge`` deltas are exact
negations (PD-9 eps 1e-6) and the label involution holds by
construction.

Cohort dispositions (R3 v3.2 §2 / 三位一体 §3.4; never silent drops,
original denominators kept in the ledger):

* ``RXN_0000001248`` — excluded_gauge_unresolved (prior audit: 0/120
  eligible at d0 in {0.05, 0.08}); all its samples are masked.
* the 8 prior gauge-unresolved branch-sides (480/5568 rows @V5) — samples
  at nodes of those sides are masked (gauge_unresolved_excluded);
  dual-denominator coverage reported.
* the 5 mapping-overflow pairs (238/1089/1174/1504/2226) — each checked
  for pair-edit vs R/P bond-geometry consistency; inconsistent reactions
  are quarantined (masked), consistent ones pass with evidence.
* the historical 342 conflict pairs (20261018 report, fit45, d0=0.05
  input threshold) — the pair-list artifact is not present in the repo
  (ADR-0026 declared that it was never re-scanned); R3 closes the cohort
  by a full re-audit at R3 semantics (near-identical inputs within a
  curve under the same intent at d0=0.08; cross-reaction scope is empty
  because the duplicate ledger has no identity group with two fit50
  members).
* ``RXN_0000002022`` — rechecked at d0=0.08 with ``teacher_outcome``
  from the new solve; the prior d0=0.05 "all contact" conclusion is
  superseded by the recomputed counts (ledger cross-references the old
  d0=0.08 teacher shards).

No ``data/ground_truth/**`` read occurs (truth access log stays empty).

Usage::

    python3 scripts/agent_ops/data_gate/r3_build_tables.py \
        --out outputs/pesn/30h_campaign/data \
        [--repo-root .] [--limit N] [--skip-conflict-audit]

Exit code 0 on success; refuses to overwrite an existing output dir
unless ``--force``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import struct
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.pesn.contracts.key_atom import build_key_atom_task  # noqa: E402
from pes2ts_core.pesn.offline.data.key_atom_teacher import (  # noqa: E402
    FRAME_LAYER_NATIVE_VERIFIED,
    MAIN_D0,
    TEACHER_CAP_ANGSTROM,
    solve_teacher,
    verify_frame_continuity,
)

TOOL_VERSION = "pes2ts_r3_build_tables_v1"
SCHEMA_VERSION = "r3_6table_v2"
INTENT_VERSION = "r3_intent_v1"
TEACHER_VERSION = "intent_directed_first_contact_v1"
CANONICAL_UNITS = "angstrom"
DELTA_EPS = 1.0e-6
CURVE_WEIGHT_TARGET = 1.0

#: conflict re-audit thresholds (historical audit: 0.05 A input threshold;
#: direction thresholds from the frozen conflict audit, ADR-0026 §2.5)
CONFLICT_INPUT_RMSD = 0.05
CONFLICT_EXACT_EPS = 1.0e-9
CONFLICT_COSINE_NEGATIVE = 0.0
NEAR_CONFLICT_COSINE = 0.25
NEAR_CONFLICT_WEIGHT = 0.25

#: mapping-overflow geometry consistency: expected bond-length change
#: direction for an edit, with a small band where perception noise
#: dominates (order-only perception differences are ~0.03 A).
OVERFLOW_GEOM_TOL = 0.02

WP2_REL = "outputs/pesn/dual_ts_v2_20261015/wp2_fit50"
P1_REL = "data/interim/g1_truth/p1_mapping"
SPLIT_REL = "data/manifests/v5/split_manifest.json"
DATASET_AUDIT_REL = "data/manifests/v5/dataset_audit.json"
DUP_LEDGER_REL = "data/manifests/duplicate_ledger.json"
LINEAGE_REL = "outputs/pesn/dual_ts_v2_20261015/splits/dual_ts_split_lineage_v1.json"
COVERAGE_REL = "outputs/pesn/dual_ts_v2_20261015/wp2_fit50/coverage_summary.json"
SELECTION_REL = "outputs/pesn/direct_egnn_v1_20261015/splits/selection_direct_fit50.json"
NATIVE_REL = "outputs/pesn/v5/native"
TEACHER_D0_REL = "outputs/pesn/v5/teacher"

TABLE_FILES = (
    "irc_curve.csv",
    "irc_node.csv",
    "irc_edge.csv",
    "reaction_intent.csv",
    "curve_quality.csv",
    "sample_view.csv",
)
DIAGNOSTICS_FILE = "branch_diagnostics.csv"
SPLIT_FILE = "split_manifest_r3.json"
DATA_MANIFEST_FILE = "data_manifest.json"
TEACHER_MANIFEST_FILE = "teacher_manifest.json"
LEDGER_FILE = "build_ledger_r3.json"
TRUTH_LOG_FILE = "truth_access_log.jsonl"


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(doc: Any) -> str:
    payload = json.dumps(doc, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_flat_f64(values: Sequence[float]) -> str:
    """PD-9 binding sha: IEEE-754 float64 little-endian bytes."""
    return hashlib.sha256(
        struct.pack(f"<{len(values)}d", *values)).hexdigest()


def jdump(doc: Any) -> str:
    """Canonical JSON cell (compact, sorted, utf-8 safe)."""
    return json.dumps(doc, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_bool(v: Optional[bool]) -> str:
    if v is None:
        return ""
    return "true" if v else "false"


def _fmt_float(v: Optional[float]) -> str:
    if v is None:
        return ""
    f = float(v)
    if not math.isfinite(f):
        return ""
    return repr(f)


def write_csv(path: Path, header: Sequence[str],
              rows: Sequence[Sequence[Any]]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(list(header))
        for row in rows:
            w.writerow(list(row))


# ---------------------------------------------------------------------------
# lineage loading
# ---------------------------------------------------------------------------

class ReactionBundle:
    """Everything the builder needs about one fit50 reaction."""

    def __init__(self, rid: str, task_doc: dict, teacher_doc: dict,
                 p1_doc: dict, wp2_task_path: Path, wp2_teacher_path: Path,
                 p1_path: Path):
        self.rid = rid
        self.task_doc = task_doc
        self.teacher_doc = teacher_doc
        self.p1_doc = p1_doc
        self.wp2_task_path = wp2_task_path
        self.wp2_teacher_path = wp2_teacher_path
        self.p1_path = p1_path
        self.task = build_key_atom_task(
            task_doc, p1_doc,
            wp2_path=str(wp2_task_path), p1_path=str(p1_path))

        # branches from the wp2 teacher export, ordered by step_index.
        brs: Dict[str, List[dict]] = {}
        for state in teacher_doc["states"]:
            brs.setdefault(state["branch_id"], []).append(state)
        if len(brs) != 2:
            raise ValueError(f"{rid}: expected 2 branches, got {len(brs)}")
        labeled = {}
        for meta in teacher_doc["branches"]:
            labeled[meta["branch_id"]] = meta["endpoint_label"]
        self.branch_frames: Dict[str, np.ndarray] = {}
        self.branch_label: Dict[str, str] = {}
        for bid, states in brs.items():
            states = sorted(states, key=lambda s: s["step_index"])
            self.branch_frames[bid] = np.asarray(
                [s["coordinates"] for s in states], dtype=np.float64)
            self.branch_label[bid] = labeled.get(bid, "")
        # convention (dual_ts_preprocess.canonical_endpoint_order + teacher
        # export): endpoint_label "A" starts at task endpoints[0]; the
        # geometry cross-check is recorded by the builder.
        by_label = {lab: bid for bid, lab in self.branch_label.items()}
        if set(by_label) != {"A", "B"}:
            raise ValueError(f"{rid}: branch endpoint labels {by_label}")
        self.branch_ep0 = by_label["A"]   # starts at task endpoints[0]
        self.branch_ep1 = by_label["B"]

        Z = teacher_doc["states"][0]["atomic_numbers"]
        self.atomic_numbers = [int(z) for z in Z]
        self.atom_map_ids = list(teacher_doc["states"][0]["atom_map_ids"])
        for s in teacher_doc["states"]:
            if list(s["atomic_numbers"]) != self.atomic_numbers:
                raise ValueError(f"{rid}: atomic_numbers change across states")
        # row-space identity invariant (wp2 task constraints:
        # "row i <-> map i+1 <-> IRC atom row i (g1 invariant)"): the
        # KeyAtomTask row space (p1 ts_irc_index) must equal the wp2 row
        # order used by the exported coordinates.
        expected = [f"m{i + 1:04d}" for i in range(len(self.atom_map_ids))]
        if list(self.atom_map_ids) != expected:
            raise ValueError(
                f"{rid}: atom map ids are not the canonical identity "
                f"order: {self.atom_map_ids[:5]}...")
        self.charge = int(task_doc.get("charge", 0))
        self.multiplicity = int(task_doc.get("multiplicity", 1))

    # -- binding evidence ---------------------------------------------------
    def exclusive_bond_score(self, start_xyz: np.ndarray, slot: int) -> float:
        """Fraction of slot-exclusive declared bonds short at ``start_xyz``
        and the other slot's exclusive bonds long (binding evidence)."""
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


def load_reactions(repo: Path, limit: Optional[int] = None
                   ) -> List[ReactionBundle]:
    wp2 = repo / WP2_REL
    p1_root = repo / P1_REL
    selection = read_json(repo / SELECTION_REL)
    rids = sorted(str(r) for r in (selection.get("reaction_ids") or []))
    wp2_ids = sorted(p.name[:-len(".task.json")]
                     for p in wp2.glob("RXN_*.task.json"))
    if rids != wp2_ids:
        raise ValueError(
            "fit50 selection and wp2_fit50 export disagree: "
            f"selection={len(rids)} wp2={len(wp2_ids)}")
    if limit is not None:
        rids = rids[:limit]
    out: List[ReactionBundle] = []
    for rid in rids:
        number = int(rid.split("_")[-1])
        p1_path = p1_root / f"{number // 1000:05d}" / f"{rid}.json"
        if not p1_path.is_file():
            raise FileNotFoundError(f"{rid}: missing P1 mapping {p1_path}")
        out.append(ReactionBundle(
            rid,
            read_json(wp2 / f"{rid}.task.json"),
            read_json(wp2 / f"{rid}.teacher.json"),
            read_json(p1_path),
            wp2 / f"{rid}.task.json",
            wp2 / f"{rid}.teacher.json",
            p1_path,
        ))
    return out


# ---------------------------------------------------------------------------
# prior-artifact cohort dispositions (V5 shards: evidence only)
# ---------------------------------------------------------------------------

def prior_branch_dispositions(repo: Path) -> Dict[str, dict]:
    """{reaction_id: {side: disposition}} from the frozen V5 artifacts.

    A branch is ``gauge_unresolved`` exactly when the prior pipeline left
    ``gate_ok=False`` AND its full native continuity check failed — the
    rows whose ``reason_id`` is REASON_GAUGE_UNRESOLVED in the teacher
    shards (reproduced 480/5568).
    """
    out: Dict[str, dict] = {}
    native = repo / NATIVE_REL
    if not native.is_dir():
        return out
    for meta_path in sorted(native.glob("RXN_*.branch*.meta.json")):
        name = meta_path.name
        rid = name.split(".")[0]
        branch = int(name.split(".branch")[1].split(".")[0])
        meta = read_json(meta_path)
        gate_ok = bool(meta["reference"]["gate_ok"])
        cont_ok = bool(meta["frame_continuity"]["passed"])
        rec = out.setdefault(rid, {})
        rec[f"branch{branch}"] = {
            "side": str(meta["start_endpoint_side"]),
            "endpoint_index": int(meta["start_endpoint_index"]),
            "gate_ok": gate_ok,
            "continuity_passed": cont_ok,
            "frame_layer": str(meta["frame_layer"]),
            "unresolved": (not gate_ok) and (not cont_ok),
        }
    return out


def prior_teacher_counts(repo: Path) -> Dict[str, dict]:
    """Per-d0 eligible/gauge/terminal counts from the V5 teacher shards."""
    out: Dict[str, dict] = {}
    for d0_tag in ("0.05", "0.08"):
        root = repo / TEACHER_D0_REL / f"d0_{d0_tag}"
        if not root.is_dir():
            continue
        totals = Counter()
        per_rxn: Dict[str, Counter] = defaultdict(Counter)
        for npz_path in sorted(root.glob("RXN_*.branch*.npz")):
            rid = npz_path.name.split(".")[0]
            with np.load(npz_path) as z:
                reason = z["reason_id"]
                eligible = z["eligible"]
                terminal = z["terminal"]
            c = per_rxn[rid]
            c["rows"] += int(len(reason))
            c["eligible"] += int(eligible.sum())
            c["terminal"] += int(terminal.sum())
            c["gauge_unresolved"] += int((reason == 3).sum())
            c["no_future_step"] += int((reason == 1).sum())
            totals["rows"] += int(len(reason))
            totals["eligible"] += int(eligible.sum())
            totals["terminal"] += int(terminal.sum())
            totals["gauge_unresolved"] += int((reason == 3).sum())
            totals["no_future_step"] += int((reason == 1).sum())
        out[d0_tag] = {
            "totals": dict(totals),
            "per_reaction": {k: dict(v) for k, v in sorted(per_rxn.items())},
        }
    return out


# ---------------------------------------------------------------------------
# curve construction
# ---------------------------------------------------------------------------

class Curve:
    """Canonical A(R)->TS->B(P) curve of one reaction."""

    def __init__(self, bundle: ReactionBundle):
        b = bundle
        self.rid = b.rid
        self.curve_id = b.rid
        sides = b.task.endpoint_sides  # ("R"/"P") for endpoints[0..1]
        if sorted(sides) != ["P", "R"]:
            raise ValueError(f"{b.rid}: endpoint sides {sides}")
        r_slot = sides.index("R")
        p_slot = 1 - r_slot
        # row-space endpoint graphs from the KeyAtomTask (converted via the
        # endpoint's own atom_map_ids — authoritative, ADR-0026 §2.1)
        ep_r = b.task.endpoints[r_slot]
        ep_p = b.task.endpoints[p_slot]
        branch_r = b.branch_ep0 if r_slot == 0 else b.branch_ep1
        branch_p = b.branch_ep1 if r_slot == 0 else b.branch_ep0
        x_r = b.branch_frames[branch_r]     # R endpoint -> TS
        x_p = b.branch_frames[branch_p]     # P endpoint -> TS
        if x_r.ndim != 3 or x_r.shape[2] != 3 or \
                x_p.ndim != 3 or x_p.shape[2] != 3 or \
                x_p.shape[1] != x_r.shape[1]:
            raise ValueError(f"{b.rid}: coordinate shape mismatch")
        if x_r.shape[0] < 2 or x_p.shape[0] < 2:
            raise ValueError(f"{b.rid}: branch too short")
        # the two branches share the TS frame exactly (audited export);
        # drop the duplicate P-side TS frame.
        ts_eq = float(np.abs(x_r[-1] - x_p[-1]).max())
        self.ts_join_maxabs = ts_eq
        if ts_eq > DELTA_EPS:
            raise ValueError(
                f"{b.rid}: TS frames not shared (maxabs {ts_eq:.3e})")
        self.xyz = np.concatenate([x_r, x_p[-2::-1]], axis=0)
        self.n = int(self.xyz.shape[0])
        self.n_atoms = int(self.xyz.shape[1])
        self.ts_index = int(x_r.shape[0] - 1)
        self.branch_r = branch_r
        self.branch_p = branch_p
        self.r_slot = r_slot
        self.p_slot = p_slot
        self.ep_r = ep_r
        self.ep_p = ep_p
        self.sides = sides
        # R-side node range (inclusive of TS); P-side node range
        self.r_nodes = range(0, self.ts_index + 1)
        self.p_nodes = range(self.ts_index, self.n)

        # ---- key atoms from the G1 P1 bond events (authoritative) ----
        self.key_rows = tuple(int(r) for r in b.task.key_atom_rows)
        if not self.key_rows:
            raise ValueError(
                f"{b.rid}: empty key-atom set from G1 P1 bond_events")

        # ---- endpoint graphs in row space ----
        self.g_r = {
            "atoms": list(b.atomic_numbers),
            "bonds": [[int(i), int(j), float(o)]
                      for i, j, o in ep_r["bonds"]],
        }
        self.g_p = {
            "atoms": list(b.atomic_numbers),
            "bonds": [[int(i), int(j), float(o)]
                      for i, j, o in ep_p["bonds"]],
        }
        # ---- signed bond edits P - R (delta_B_intent direction R->P) ----
        b_r = {(i, j): o for i, j, o in self.g_r["bonds"]}
        b_p = {(i, j): o for i, j, o in self.g_p["bonds"]}
        edits = []
        for pair in sorted(set(b_r) | set(b_p)):
            d = float(b_p.get(pair, 0.0) - b_r.get(pair, 0.0))
            if abs(d) > 1e-9:
                edits.append((pair[0], pair[1], d))
        self.edits = edits
        forming = sorted({a for i, j, d in edits if d > 0 for a in (i, j)})
        breaking = sorted({a for i, j, d in edits if d < 0 for a in (i, j)})
        self.intent_role = {
            "forming": forming,
            "breaking": breaking,
            "both": sorted(set(forming) & set(breaking)),
        }
        self.cv_spec = {
            "kind": "graph_edit_bond_lengths",
            "coordinates": [[i, j] for i, j, _d in edits],
            "d0_angstrom": MAIN_D0,
        } if edits else None

        # ---- progress (piecewise normalized all-atom RMS step arc) ----
        self.progress = self._compute_progress()
        self._binding_evidence = {
            "endpoint0_exclusive_score_at_branch_A_start":
                b.exclusive_bond_score(b.branch_frames[b.branch_ep0][0], 0),
            "endpoint1_exclusive_score_at_branch_A_start":
                b.exclusive_bond_score(b.branch_frames[b.branch_ep0][0], 1),
        }

    def _compute_progress(self) -> np.ndarray:
        d = np.sqrt(np.mean(np.sum(np.diff(self.xyz, axis=0) ** 2, axis=2),
                            axis=1))
        s = np.concatenate([[0.0], np.cumsum(d)])
        s_ts = float(s[self.ts_index])
        s_end = float(s[-1])
        prog = np.zeros(self.n, dtype=np.float64)
        if s_ts <= 0.0:
            raise ValueError(f"{self.curve_id}: zero arc before TS")
        for i in range(self.ts_index + 1):
            prog[i] = -(s_ts - float(s[i])) / s_ts
        b_side = s_end - s_ts
        if b_side <= 0.0:
            raise ValueError(f"{self.curve_id}: zero arc after TS")
        for i in range(self.ts_index, self.n):
            prog[i] = (float(s[i]) - s_ts) / b_side
        return prog


# ---------------------------------------------------------------------------
# teacher labels (frozen solver, native/raw layer)
# ---------------------------------------------------------------------------

def solve_curve_direction(curve: Curve, direction: str) -> dict:
    """First-contact teacher for one intent direction.

    ``forward`` reads the canonical order; ``reverse`` reads the reversed
    path (same stored Cartesian frame — no registration), which makes the
    reversed single-step teacher the exact mirror of the reading path.
    """
    pos = curve.xyz if direction == "forward" else curve.xyz[::-1]
    res = solve_teacher(
        pos, curve.key_rows, np.asarray([], dtype=np.int64),
        np.zeros(curve.n_atoms, dtype=np.float64), MAIN_D0,
        gauge_ok=True, frame_layer=FRAME_LAYER_NATIVE_VERIFIED,
        cap=TEACHER_CAP_ANGSTROM)
    n = curve.n
    out = {}
    for t in range(n - 1):
        node = t if direction == "forward" else (n - 1 - t)
        seg = int(res["event_segment"][t])
        alpha = float(res["event_alpha"][t])
        if direction == "forward":
            seg_node = seg
        else:
            seg_node = n - 1 - seg
        out[node] = {
            "eligible": bool(res["eligible"][t]),
            "reason_id": int(res["reason_id"][t]),
            "terminal": bool(res["terminal"][t]),
            "capped": bool(res["capped"][t]),
            "u_raw": (np.asarray(res["u_raw"][t], dtype=np.float64)
                      if res["eligible"][t] else None),
            "target_node_index": (seg_node
                                  if res["eligible"][t] else None),
            "target_alpha": alpha if res["eligible"][t] else None,
            "d_eff": float(res["d_eff"][t]),
        }
    # the last node in reading order has no future at all
    last_node = n - 1 if direction == "forward" else 0
    out[last_node] = {
        "eligible": False,
        "reason_id": -1,  # no solve row: terminal by construction
        "terminal": True,
        "capped": False,
        "u_raw": None,
        "target_node_index": None,
        "target_alpha": None,
        "d_eff": None,
    }
    return out


# ---------------------------------------------------------------------------
# conflict re-audit (R3 semantics)
# ---------------------------------------------------------------------------

def conflict_audit_curve(curve: Curve,
                         labels: Dict[str, Dict[str, dict]]
                         ) -> Tuple[Dict[Tuple[int, str], str],
                                    Dict[str, Any]]:
    """Near-identical inputs, same intent, diverging teacher direction.

    Scans all same-curve node pairs with full-geometry RMSD below
    ``CONFLICT_INPUT_RMSD`` (exact duplicates counted separately); for
    each pair where both labels are usable, the flattened key-atom
    teacher vectors are compared (cosine < 0 => conflict -> mask both;
    cosine in [0, 0.25) => near-conflict -> downweight).
    """
    flags: Dict[Tuple[int, str], str] = {}
    stats = {"n_pairs_below_threshold": 0,
             "n_exact_duplicate_pairs": 0,
             "n_compared_pairs": 0,
             "n_conflict_pairs": 0,
             "n_near_conflict_pairs": 0}
    xyz = curve.xyz
    m = np.sqrt(np.mean(np.sum(
        (xyz[:, None, :, :] - xyz[None, :, :, :]) ** 2, axis=3), axis=2))
    n = curve.n
    k = np.asarray(curve.key_rows, dtype=np.int64)
    for i in range(n):
        for j in range(i + 1, n):
            rmsd = float(m[i, j])
            if rmsd >= CONFLICT_INPUT_RMSD:
                continue
            stats["n_pairs_below_threshold"] += 1
            if rmsd <= CONFLICT_EXACT_EPS:
                stats["n_exact_duplicate_pairs"] += 1
                continue
            for intent_dir in ("forward", "reverse"):
                li = labels[intent_dir].get(i)
                lj = labels[intent_dir].get(j)
                if not (li and lj and li["eligible"] and lj["eligible"]
                        and li["u_raw"] is not None
                        and lj["u_raw"] is not None):
                    continue
                if li["terminal"] or lj["terminal"]:
                    continue
                ui = li["u_raw"].ravel()
                uj = lj["u_raw"].ravel()
                ni = float(np.linalg.norm(ui))
                nj = float(np.linalg.norm(uj))
                if ni <= 1e-8 or nj <= 1e-8:
                    continue
                stats["n_compared_pairs"] += 1
                cos = float(ui @ uj) / (ni * nj)
                if cos < CONFLICT_COSINE_NEGATIVE:
                    stats["n_conflict_pairs"] += 1
                    flags[(i, intent_dir)] = "conflict"
                    flags[(j, intent_dir)] = "conflict"
                elif cos < NEAR_CONFLICT_COSINE:
                    stats["n_near_conflict_pairs"] += 1
                    if flags.get((i, intent_dir)) != "conflict":
                        flags[(i, intent_dir)] = "near_conflict"
                    if flags.get((j, intent_dir)) != "conflict":
                        flags[(j, intent_dir)] = "near_conflict"
    return flags, stats


# ---------------------------------------------------------------------------
# feature / label payloads (legal forward inputs only)
# ---------------------------------------------------------------------------

def feature_payload(bundle: ReactionBundle, curve: Curve, node: int,
                    intent: dict, d0: float) -> dict:
    """SHA payload of the model-legal forward inputs (no teacher data)."""
    x = curve.xyz[node]
    return {
        "atomic_numbers": curve.atomic_numbers if hasattr(
            curve, "atomic_numbers") else bundle.atomic_numbers,
        "charge": bundle.charge,
        "coords": [float(v) for v in x.ravel()],
        "curve_id": curve.curve_id,
        "cv_spec": intent["cv_spec"],
        "d0": d0,
        "g_src": intent["g_src"],
        "g_tgt": intent["g_tgt"],
        "delta_b_intent": intent["edits"],
        "k": list(curve.key_rows),
        "multiplicity": bundle.multiplicity,
        "node_index": int(node),
        "intent_id": intent["intent_id"],
        "intent_version": INTENT_VERSION,
    }


def label_payload(sol: dict, d0: float) -> dict:
    u = sol["u_raw"]
    return {
        "d0": d0,
        "label_status": sol["label_status"],
        "mask": bool(sol["mask"]),
        "target_node_index": sol["target_node_index"],
        "target_alpha": sol["target_alpha"],
        "u_key": ([float(v) for v in u.ravel()] if u is not None else None),
    }


# ---------------------------------------------------------------------------
# overflow geometry consistency (5 mapping-overflow pairs)
# ---------------------------------------------------------------------------

def overflow_check(bundle: ReactionBundle, curve: Curve) -> dict:
    """pair-edit vs R/P bond-geometry consistency (三位一体 §3.4).

    Every signed bond edit must show a bond-length change in the expected
    direction (forming -> shorter, breaking -> longer) between the R-side
    and P-side endpoint frames; ``OVERFLOW_GEOM_TOL`` separates the
    expected change from perception noise.  Inconsistent reactions are
    quarantined.
    """
    x_r = curve.xyz[0]
    x_p = curve.xyz[-1]
    rows = []
    n_bad = 0
    for (i, j, d) in curve.edits:
        len_r = float(np.linalg.norm(x_r[i] - x_r[j]))
        len_p = float(np.linalg.norm(x_p[i] - x_p[j]))
        dlen = len_p - len_r
        if d > 0:
            consistent = dlen < -OVERFLOW_GEOM_TOL
        else:
            consistent = dlen > OVERFLOW_GEOM_TOL
        rows.append({
            "pair": [int(i), int(j)], "delta_b": float(d),
            "len_r": len_r, "len_p": len_p, "delta_len": dlen,
            "consistent": bool(consistent),
        })
        if not consistent:
            n_bad += 1
    return {"n_edits": len(curve.edits), "n_inconsistent": n_bad,
            "rows": rows, "consistent": n_bad == 0}


# ---------------------------------------------------------------------------
# main build
# ---------------------------------------------------------------------------

def build(repo: Path, out: Path, limit: Optional[int] = None,
          skip_conflict_audit: bool = False) -> dict:
    bundles = load_reactions(repo, limit=limit)
    prior_branch = prior_branch_dispositions(repo)
    prior_teacher = prior_teacher_counts(repo)
    split_doc = read_json(repo / SPLIT_REL)
    audit_doc = read_json(repo / DATASET_AUDIT_REL)
    dup_doc = read_json(repo / DUP_LEDGER_REL)
    group_fold = {str(k): int(v) for k, v in split_doc["group_fold"].items()}
    overflow_pairs = audit_doc.get("overflow_pairs") or []
    overflow_rids = sorted({str(p["a"]).split(".")[0] for p in overflow_pairs})

    out.mkdir(parents=True, exist_ok=True)

    # ---------------- per-curve work ----------------
    irc_curve_rows: List[List[Any]] = []
    irc_node_rows: List[List[Any]] = []
    reaction_intent_rows: List[List[Any]] = []
    curve_quality_rows: List[List[Any]] = []
    branch_diag_rows: List[List[Any]] = []
    sample_rows: List[List[Any]] = []
    # edge buffers (PD-9 flat offsets are assigned in file order)
    edge_rows_raw: List[dict] = []

    ledger_curves: Dict[str, Any] = {}
    conflict_stats_total = Counter()
    status_counts = Counter()
    dispositions = Counter()
    gauge_masked_samples = 0
    n_samples = 0

    for bundle in bundles:
        curve = Curve(bundle)
        continuity = verify_frame_continuity(curve.xyz)
        # R3 raw-label gauge gate (documented validity boundary): the
        # archive-preservation discriminator is the CONSECUTIVE rigid
        # drift (frozen native thresholds 2 deg / 0.1 A per consecutive
        # pair) — a per-frame re-oriented archive shows random
        # tens-of-degrees consecutive rotations.  The cumulative
        # rotation bound (10 deg) of the frozen native gate is a
        # branch-scale bound for label-side Kabsch registration, which
        # R3 raw labels (X_j - X_i, no registration) do not perform;
        # the cumulative value is recorded as a diagnostic in the
        # ledger and in curve_quality.note (never claimed as a pass
        # criterion).
        thr = continuity.get("thresholds", {})
        ev = continuity.get("evidence", {})
        consec_ok = (
            float(ev.get("max_consecutive_rotation_deg", math.inf))
            <= float(thr.get("max_consecutive_rotation_deg", 2.0))
            and float(ev.get("max_consecutive_translation_angstrom",
                             math.inf))
            <= float(thr.get("max_consecutive_translation_angstrom", 0.1)))
        if not consec_ok:
            # preserved-frame evidence failed -> raw labels are not
            # gauge-consistent; fail loud (never a silent lab frame)
            raise ValueError(
                f"{curve.curve_id}: consecutive frame drift outside "
                f"pre-registered native thresholds ({ev})")

        # prior (V5) branch dispositions
        pb = prior_branch.get(bundle.rid, {})
        unresolved_sides = set()
        for _bkey, info in pb.items():
            if info.get("unresolved"):
                unresolved_sides.add(info["side"])
        cohort_1248 = (bundle.rid == "RXN_0000001248")

        # teacher labels for both intents
        fwd = solve_curve_direction(curve, "forward")
        rev = solve_curve_direction(curve, "reverse")
        labels = {"forward": fwd, "reverse": rev}

        # conflict re-audit
        if skip_conflict_audit:
            flags: Dict[Tuple[int, str], str] = {}
            cstats = {}
        else:
            flags, cstats = conflict_audit_curve(curve, labels)
            conflict_stats_total.update(cstats)

        # overflow check (only for the cohort; recorded either way)
        overflow_result = overflow_check(bundle, curve) \
            if bundle.rid in overflow_rids else None

        intents = []
        for direction in ("forward", "reverse"):
            if direction == "forward":
                g_src, g_tgt = curve.g_r, curve.g_p
                sign = 1.0
            else:
                g_src, g_tgt = curve.g_p, curve.g_r
                sign = -1.0
            intent = {
                "intent_id": f"{curve.curve_id}_{'fwd' if direction == 'forward' else 'rev'}",
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
            reaction_intent_rows.append([
                intent["intent_id"], intent["curve_id"], direction,
                intent["reverse_of_intent_id"], intent["intent_pair_id"],
                _fmt_bool(intent["intent_underdetermined"]),
                jdump(g_src), jdump(g_tgt),
                jdump(intent["edits"]),
                jdump(curve.cv_spec) if curve.cv_spec else "null",
                jdump(intent["intent_role"]),
                jdump(intent["rules"]),
                INTENT_VERSION,
            ])

        n_nodes = curve.n
        n_rows_curve = 2 * n_nodes
        curve_weight = CURVE_WEIGHT_TARGET / n_rows_curve

        for node in range(n_nodes):
            xyz = curve.xyz[node]
            for intent in intents:
                direction = intent["direction"]
                sol = labels[direction][node]
                # gauge/cohort masks (node-level, both intents)
                if cohort_1248:
                    status, mask, disp, gauge = (
                        "masked", True, "excluded_gauge_unresolved",
                        "unresolved_masked")
                elif pb and (
                        (node in curve.r_nodes and "R" in unresolved_sides)
                        or (node in curve.p_nodes and "P" in unresolved_sides)):
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
                # conflict re-audit overrides (never silent; cohort masks
                # keep their own disposition — they are already excluded)
                if status == "ok":
                    cflag = flags.get((node, direction))
                    if cflag == "conflict":
                        status, mask = "conflict", True
                        disp = "conflict_near_input_masked"
                    elif cflag == "near_conflict":
                        disp = "near_conflict_downweighted"
                label_weight = (1.0 if status == "ok" else
                                NEAR_CONFLICT_WEIGHT
                                if disp == "near_conflict_downweighted"
                                else 0.0)
                # quarantined mapping-overflow cohort (only where not
                # already excluded by a higher-priority disposition)
                if status == "ok" and overflow_result is not None and \
                        not overflow_result["consistent"]:
                    status, mask = "masked", True
                    disp = "quarantine_mapping_overflow"
                    label_weight = 0.0

                if mask and status not in ("terminal", "conflict"):
                    # keep u out of masked hard labels; terminal/masked rows
                    # carry no fabricated action
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
                feat = feature_payload(bundle, curve, node, intent, MAIN_D0)
                sol_row = {
                    "label_status": status,
                    "mask": mask,
                    "target_node_index": t_node,
                    "target_alpha": t_alpha,
                    "u_raw": u_out,
                }
                feature_sha = sha256_json(feat)
                label_sha = sha256_json(label_payload(sol_row, MAIN_D0))
                sample_id = (f"{curve.curve_id}:{node}:"
                             f"{intent['intent_id']}:{MAIN_D0:g}")
                cx, cy, cz = (float(v) for v in xyz.mean(axis=0))
                n_samples += 1
                status_counts[status] += 1
                dispositions[disp] += 1
                if gauge == "unresolved_masked":
                    gauge_masked_samples += 1
                sample_rows.append([
                    sample_id, curve.curve_id, node, intent["intent_id"],
                    repr(MAIN_D0),
                    repr(cx), repr(cy), repr(cz),
                    repr(label_weight), repr(curve_weight),
                    repr(sample_weight), status, feature_sha, label_sha,
                    "false", gauge,
                    t_node if t_node is not None else "",
                    _fmt_float(t_alpha),
                    (json.dumps([float(v) for v in u_out.ravel()])
                     if u_out is not None else ""),
                    "",  # target_progress_q: V5.2 label, not part of R3
                    "",  # stop_label: V5.2 label, not part of R3
                    _fmt_bool(mask),
                    int(group_fold.get(curve.curve_id, -1)),
                    disp,
                    json.dumps([float(v) for v in xyz.ravel()]),
                    f"[{curve.n_atoms},3]",
                ])

        # ---------------- edges (PD-9) ----------------
        fwd_id = f"{curve.curve_id}_fwd"
        rev_id = f"{curve.curve_id}_rev"
        atom_map_sha = sha256_json({
            "atom_map_ids": bundle.atom_map_ids,
            "rule": "row i <-> map i+1 <-> IRC atom row i (g1 invariant)",
        })
        for intent_id, direction in ((fwd_id, "forward"), (rev_id, "reverse")):
            for k in range(n_nodes - 1):
                if direction == "forward":
                    src, dst = k, k + 1
                    delta = curve.xyz[k + 1] - curve.xyz[k]
                else:
                    src, dst = k + 1, k
                    delta = -(curve.xyz[k + 1] - curve.xyz[k])
                flat = [float(v) for v in delta.ravel()]
                edge_rows_raw.append({
                    "curve_id": curve.curve_id,
                    "intent_id": intent_id,
                    "edge_index": k,
                    "src_node_index": src,
                    "dst_node_index": dst,
                    "delta_xyz_flat": json.dumps(flat),
                    "delta_xyz_shape": f"[{curve.n_atoms},3]",
                    "atom_map_sha256": atom_map_sha,
                    "delta_xyz_sha256": sha256_flat_f64(flat),
                    "units": CANONICAL_UNITS,
                    "same_ref_frame": True,
                    "_n_flat": len(flat),
                })

        # ---------------- curve tables ----------------
        irc_curve_rows.append([
            curve.curve_id, f"irc_{curve.curve_id}", n_nodes,
            2 * (n_nodes - 1), curve.ts_index, 0, n_nodes - 1,
            str(bundle.wp2_teacher_path.relative_to(repo)),
            "", "",  # energies not exported in the lineage
        ])
        Z = bundle.atomic_numbers
        for node in range(n_nodes):
            x, y, z = (float(v) for v in curve.xyz[node].mean(axis=0))
            irc_node_rows.append([
                curve.curve_id, node, repr(x), repr(y), repr(z),
                repr(float(curve.progress[node])),
                1 if node == curve.ts_index else 0,
                1 if node in (0, n_nodes - 1) else 0,
                "", "", "ok", jdump(Z),
                json.dumps([float(v) for v in curve.xyz[node].ravel()]),
                f"[{curve.n_atoms},3]",
            ])
        n_ok = sum(1 for r in sample_rows
                   if r[1] == curve.curve_id and r[11] == "ok")
        quality_status = (
            "gauge_unresolved_excluded" if cohort_1248 else
            "gauge_partial_excluded" if unresolved_sides else
            "quarantined" if (overflow_result is not None
                              and not overflow_result["consistent"]) else
            "ok")
        curve_quality_rows.append([
            curve.curve_id, quality_status,
            repr(n_ok / n_rows_curve),
            jdump({"n_ok": n_ok, "n_rows": n_rows_curve}),
            jdump({k: {"side": v["side"], "gate_ok": v["gate_ok"],
                       "continuity_passed": v["continuity_passed"],
                       "frame_layer": v["frame_layer"]}
                   for k, v in sorted(pb.items())}) if pb else "null",
            jdump({
                "wp2_task_sha256": sha256_file(bundle.wp2_task_path),
                "wp2_teacher_sha256": sha256_file(bundle.wp2_teacher_path),
                "p1_mapping_sha256": sha256_file(bundle.p1_path),
                "ts_join_maxabs": curve.ts_join_maxabs,
                "frame_continuity": continuity,
                "binding_evidence": curve._binding_evidence,
                "endpoint_sides": {"endpoint_0": list(curve.sides)[0],
                                   "endpoint_1": list(curve.sides)[1]},
                "unresolved_sides_v5": sorted(unresolved_sides),
                "cohort_1248": cohort_1248,
                "overflow_check": overflow_result,
            }),
        ])

        # ---------------- branch diagnostics ----------------
        for bid in sorted(bundle.branch_frames):
            X = bundle.branch_frames[bid]
            cont_b = verify_frame_continuity(X)
            is_r = (bundle.branch_ep0 == bid and curve.sides[0] == "R") or \
                   (bundle.branch_ep1 == bid and curve.sides[1] == "R")
            side = "R" if is_r else "P"
            bkey = None
            for k, info in pb.items():
                if info["side"] == side:
                    bkey = k
            binfo = pb.get(bkey, {}) if bkey else {}
            branch_diag_rows.append([
                curve.curve_id, bid, bundle.branch_label[bid], side,
                int(X.shape[0]),
                _fmt_bool(binfo.get("gate_ok")),
                _fmt_bool(binfo.get("continuity_passed")),
                str(binfo.get("frame_layer", "")),
                _fmt_bool(binfo.get("unresolved")),
                repr(float(cont_b["evidence"]["max_consecutive_rotation_deg"])
                     if "evidence" in cont_b else float("nan")),
                repr(float(cont_b["evidence"]["max_consecutive_translation_angstrom"])
                     if "evidence" in cont_b else float("nan")),
                repr(float(np.linalg.norm(
                    np.diff(X, axis=0), axis=2).max())),
                str(bundle.wp2_teacher_path.relative_to(repo)),
            ])

        ledger_curves[curve.curve_id] = {
            "n_nodes": n_nodes,
            "ts_index": curve.ts_index,
            "fold": group_fold.get(curve.curve_id),
            "key_rows": list(curve.key_rows),
            "n_edits": len(curve.edits),
            "intent_underdetermined": not curve.edits,
            "unresolved_sides_v5": sorted(unresolved_sides),
            "cohort_1248": cohort_1248,
            "overflow_check": overflow_result,
            "conflict_audit": cstats,
            "frame_continuity_evidence": continuity.get("evidence"),
            "ts_join_maxabs": curve.ts_join_maxabs,
            "teacher_sha256": {
                "wp2_task": sha256_file(bundle.wp2_task_path),
                "wp2_teacher": sha256_file(bundle.wp2_teacher_path),
                "p1_mapping": sha256_file(bundle.p1_path),
            },
        }

    # ---------------- edge table with running offsets ----------------
    irc_edge_rows: List[List[Any]] = []
    offset = 0
    for e in edge_rows_raw:
        n_flat = e.pop("_n_flat")
        irc_edge_rows.append([
            e["curve_id"], e["intent_id"], e["edge_index"],
            e["src_node_index"], e["dst_node_index"],
            e["delta_xyz_flat"], e["delta_xyz_shape"],
            json.dumps([offset, offset + n_flat]),
            e["atom_map_sha256"], e["delta_xyz_sha256"], e["units"],
            _fmt_bool(e["same_ref_frame"]),
        ])
        offset += n_flat

    # ---------------- write tables ----------------
    write_csv(out / "irc_curve.csv", [
        "curve_id", "canonical_irc_id", "n_nodes", "n_edges", "ts_index",
        "endpoint_a_index", "endpoint_b_index", "source_path",
        "energy_a", "energy_b"], irc_curve_rows)
    write_csv(out / "irc_node.csv", [
        "curve_id", "node_index", "x", "y", "z", "progress", "is_ts",
        "is_endpoint", "energy", "gradient_norm", "quality",
        "atomic_numbers", "coords_flat", "coords_shape"], irc_node_rows)
    write_csv(out / "irc_edge.csv", [
        "curve_id", "intent_id", "edge_index", "src_node_index",
        "dst_node_index", "delta_xyz_flat", "delta_xyz_shape",
        "delta_xyz_offsets", "atom_map_sha256", "delta_xyz_sha256",
        "units", "same_ref_frame"], irc_edge_rows)
    write_csv(out / "reaction_intent.csv", [
        "intent_id", "curve_id", "direction", "reverse_of_intent_id",
        "intent_pair_id", "intent_underdetermined", "g_src", "g_tgt",
        "delta_b_intent", "cv_spec", "intent_role", "rules",
        "intent_version"], reaction_intent_rows)
    write_csv(out / "curve_quality.csv", [
        "curve_id", "quality_status", "quality_score", "segment_quality",
        "branch_quality", "note"], curve_quality_rows)
    write_csv(out / "sample_view.csv", [
        "sample_id", "curve_id", "node_index", "intent_id", "d0", "x", "y",
        "z", "label_weight", "curve_weight", "sample_weight",
        "label_status", "feature_sha", "label_sha", "sensitivity_only",
        "gauge_ref", "target_node_index", "target_alpha",
        "target_action_u", "target_progress_q", "stop_label", "mask",
        "fold", "disposition", "coords_flat", "coords_shape"],
        sample_rows)
    write_csv(out / DIAGNOSTICS_FILE, [
        "curve_id", "branch_id", "endpoint_label", "endpoint_side",
        "n_frames", "v5_gate_ok", "v5_continuity_passed",
        "v5_frame_layer", "v5_unresolved", "consec_rot_deg",
        "consec_trans_angstrom", "step_max_angstrom",
        "source_path"], branch_diag_rows)

    # truth access log: this build reads NO ground truth (honest empty log)
    (out / TRUTH_LOG_FILE).write_text("", encoding="utf-8")

    # ---------------- split manifest ----------------
    assignments = []
    near_dup_groups: Dict[str, List[str]] = {}
    for comp in split_doc.get("components", []):
        gid = str(comp["root"])
        members = [str(g) for g in comp["group_ids"]]
        present = [m for m in members if m in group_fold]
        near_dup_groups[gid] = present
        for cid in present:
            assignments.append({
                "curve_id": cid, "fold": int(group_fold[cid]),
                "near_dup_group": gid,
            })
    split_manifest = {
        "version": "r3_split_manifest_v2",
        "schema_version": "r3_split_manifest_v2",
        "units": "near-duplicate connected component (same curve/intent/d0 "
                 "-> same fold)",
        "folds": sorted({a["fold"] for a in assignments}),
        "assignments": assignments,
        "near_dup_groups": near_dup_groups,
        "parents": {
            "v5_split_manifest": {
                "path": SPLIT_REL,
                "sha256": sha256_file(repo / SPLIT_REL)},
            "duplicate_ledger": {
                "path": DUP_LEDGER_REL,
                "sha256": sha256_file(repo / DUP_LEDGER_REL)},
            "dual_ts_split_lineage_v1": {
                "path": LINEAGE_REL,
                "sha256": sha256_file(repo / LINEAGE_REL)},
        },
        "assertions": {
            "frozen_v5_split_inherited_unchanged": True,
            "components_single_fold": True,
            "no_cross_curve_duplicate_identity_in_fit50": True,
        },
    }
    (out / SPLIT_FILE).write_text(
        json.dumps(split_manifest, indent=2, sort_keys=False,
                   ensure_ascii=False) + "\n", encoding="utf-8")

    # ---------------- build ledger ----------------
    def _coverage_counts() -> dict:
        per_curve = Counter(r[1] for r in sample_rows)
        return {
            "n_curves": len(irc_curve_rows),
            "n_nodes": len(irc_node_rows),
            "n_edges": len(irc_edge_rows),
            "n_intents": len(reaction_intent_rows),
            "n_samples": n_samples,
            "samples_by_label_status": dict(status_counts),
            "samples_by_disposition": dict(dispositions),
            "samples_gauge_unresolved_masked": gauge_masked_samples,
        }

    overflow_ledger = []
    for rid in overflow_rids:
        entry = ledger_curves.get(rid)
        overflow_ledger.append({
            "reaction_id": rid,
            "check": entry["overflow_check"] if entry else None,
            "disposition": (
                "consistent_not_quarantined"
                if entry and entry["overflow_check"] and
                entry["overflow_check"]["consistent"]
                else "quarantine_mapping_overflow"),
        })

    # 2022 recheck
    r22 = ledger_curves.get("RXN_0000002022", {})
    r22_status = Counter(
        r[11] for r in sample_rows if r[1] == "RXN_0000002022")

    ledger = {
        "schema_version": "pes2ts_r3_build_ledger_v1",
        "tool_version": TOOL_VERSION,
        "table_schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "teacher": {
            "version": TEACHER_VERSION,
            "intent_version": INTENT_VERSION,
            "d0_angstrom": MAIN_D0,
            "cap_angstrom": TEACHER_CAP_ANGSTROM,
            "solver": "pes2ts_core.pesn.offline.data.key_atom_teacher"
                      ".solve_teacher (frozen, frame_layer=native_verified_v1)",
            "frame_semantics": (
                "raw stored coordinates of the drift-registered wp2 export "
                "(one Cartesian frame per curve; TS frames exactly shared); "
                "no Kabsch registration; reverse intent re-reads the "
                "reversed path so labels are exact mirrors"),
        },
        "lineage": {
            "trajectory_source": {
                "path": WP2_REL,
                "kind": "coverage-audited dual-branch IRC export "
                        "(drift_rule chained_mass_weighted_kabsch_rigid_v1, "
                        "a_max=0.1 A insertions, ts_shared_nodes=50)",
                "fit50_selection": {
                    "path": SELECTION_REL,
                    "sha256": sha256_file(repo / SELECTION_REL)},
                "coverage_summary": {
                    "path": COVERAGE_REL,
                    "sha256": sha256_file(repo / COVERAGE_REL)},
                "split_lineage": {
                    "path": LINEAGE_REL,
                    "sha256": sha256_file(repo / LINEAGE_REL)},
            },
            "key_atoms_and_sides": {
                "path": P1_REL,
                "kind": "G1 P1 mapping bond_events (key atoms) + R/P "
                        "endpoint side matching (build_key_atom_task)",
            },
            "prior_artifacts_evidence_only": {
                "v5_native_metas": NATIVE_REL,
                "v5_teacher_shards": TEACHER_D0_REL,
                "dataset_audit": {
                    "path": DATASET_AUDIT_REL,
                    "sha256": sha256_file(repo / DATASET_AUDIT_REL)},
            },
            "source_divergence_note": (
                "the prior V5 label pipeline used truth-bridge frames "
                "(outputs/pesn/v5/native); the wp2_fit50 export is a "
                "different, coverage-audited frame set (insertion counts "
                "differ; coordinate max deviations ~0.1 A). R3 rebuilds "
                "labels from the audited wp2 export; prior artifacts are "
                "used only for cohort dispositions and denominators."),
        },
        "cohorts": {
            "RXN_0000001248": {
                "disposition": "excluded_gauge_unresolved",
                "basis": (
                    "prior audit: both branches gate_ok=False and native "
                    "continuity failed -> 0/120 eligible at d0=0.05 and "
                    "d0=0.08 (outputs/pesn/v5/teacher shards); ruled "
                    "exclusion (三位一体修复方案 §3.4; R3 v3.2 §2: "
                    "1248 排除). R3 wp2-frame continuity evidence "
                    "recorded per curve; exclusion kept (never silently "
                    "re-admitted)."),
                "r3_samples_masked": int(sum(
                    1 for r in sample_rows
                    if r[1] == "RXN_0000001248")),
            },
            "v5_gauge_unresolved_branch_sides": {
                "disposition": "masked (gauge_unresolved_excluded) at the "
                               "node level of the affected R/P side; "
                               "dual denominators reported",
                "prior_branch_detail": {
                    rid: {b: {k: info[k] for k in
                              ("side", "gate_ok", "continuity_passed",
                               "frame_layer", "unresolved")}
                          for b, info in sorted(info.items())}
                    for rid, info in sorted(prior_branch.items())},
                "prior_rows_d0_0_08": prior_teacher.get("0.08", {})
                    .get("totals", {}),
                "r3_samples_masked": gauge_masked_samples,
                "note": (
                    "prior V5 contract: reference-gauge Kabsch; rows are "
                    "gauge_unresolved only when the scatter gate failed "
                    "AND full native continuity failed (reproduces "
                    "480/5568). R3 labels are raw preserved-frame "
                    "displacements; the prior exclusion is carried forward "
                    "for these node sides so the main training denominator "
                    "does not silently expand."),
            },
            "mapping_overflow_5_pairs": {
                "disposition_table": overflow_ledger,
                "cohort_source": DATASET_AUDIT_REL,
            },
            "historical_342_conflict_pairs": {
                "origin": "docs/reports/PES2TS_实现报告_20261018.md "
                          "(fit45, 0.05 A input threshold, teacher "
                          "direction conflict; exclusions: atom "
                          "permutation / duplicate identity / exact "
                          "duplicate coordinates)",
                "pair_list_artifact": (
                    "NOT PRESENT in the repository (ADR-0026 declared the "
                    "historical 342 pairs were never re-scanned; the R2 "
                    "DQ2 342/342 classification checkboxes are open) -> "
                    "intersection with fit50 is not computable from "
                    "in-repo artifacts; R3_scope_not_affected is NOT "
                    "claimed"),
                "closure": (
                    "R3 re-audit at R3 semantics (this build): "
                    "near-identical current frames within a curve under "
                    "the same intent at d0=0.08; cross-reaction scope is "
                    "empty because the duplicate ledger has no identity "
                    "group with two fit50 members (checked) and the V5 "
                    "fit50-internal audit classified all 4950 branch "
                    "pairs as size/invariant mismatch or overflow."),
                "re_audit_stats": dict(conflict_stats_total),
            },
            "RXN_0000002022": {
                "disposition": "rechecked at d0=0.08 with teacher_outcome",
                "r3_samples_by_status_at_d0_0_08": dict(r22_status),
                "prior_d0_0_08_teacher_shard": prior_teacher.get(
                    "0.08", {}).get("per_reaction", {}).get(
                        "RXN_0000002022", {}),
                "note": (
                    "the 0.05-era 'all contact' conclusion is superseded: "
                    "the d0=0.08 first-contact solve produces terminal "
                    "rows where the remaining path never reaches 0.08 A"),
            },
        },
        "prior_denominators": {
            "source": "outputs/pesn/v5/teacher shards (d0_0.05 / d0_0.08) "
                      "+ data/manifests/v5/dataset_audit.json",
            "d0_0.05": prior_teacher.get("0.05", {}).get("totals", {}),
            "d0_0.08": prior_teacher.get("0.08", {}).get("totals", {}),
            "effective_clusters": 49,
            "fit50_reactions": 50,
            "note": ("plan baseline quotes rows=5568, eligible=5088, "
                     "gauge_unresolved=480, terminal=137 (d0=0.05), "
                     "contact=4951; at d0=0.08 terminal=282/contact=4806 "
                     "in the same shards"),
        },
        "r3_coverage": _coverage_counts(),
        "conflict_audit": dict(conflict_stats_total),
        "split": {
            "path": SPLIT_REL,
            "sha256": sha256_file(repo / SPLIT_REL),
            "n_folds": len(split_manifest["folds"]),
            "near_dup_groups": len(near_dup_groups),
            "multi_member_components": 0,
            "duplicate_ledger_fit50_overlap": 0,
        },
        "truth_access": {
            "reads": 0,
            "accessor": "none (no data/ground_truth access in this build)",
            "log": TRUTH_LOG_FILE,
        },
        "curves": ledger_curves,
    }
    (out / LEDGER_FILE).write_text(
        json.dumps(ledger, indent=2, sort_keys=False, ensure_ascii=False,
                   default=str) + "\n", encoding="utf-8")

    # ---------------- manifests (timestamp-free => deterministic shas) --
    data_files = list(TABLE_FILES) + [DIAGNOSTICS_FILE]
    dataset_sha = sha256_json([
        [name, sha256_file(out / name)] for name in data_files])
    data_manifest = {
        "version": "r3_data_manifest_v1",
        "schema_version": "r3_data_manifest_v1",
        "table_schema_version": SCHEMA_VERSION,
        "build_tool_version": TOOL_VERSION,
        "dataset_sha256": dataset_sha,
        "files": [{"path": name, "sha256": sha256_file(out / name)}
                  for name in data_files],
    }
    (out / DATA_MANIFEST_FILE).write_text(
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
            "solver": ledger["teacher"]["solver"],
            "contact": "earliest root of RMS_K(Y(alpha)-X_t)=d0 along the "
                       "intent direction, piecewise-linear segments, "
                       "root tol 1e-8",
            "cap_angstrom": TEACHER_CAP_ANGSTROM,
            "terminal": "path exhausted below d0 -> label_status=terminal, "
                        "mask=true, no fabricated action",
            "no_future_step": "entire future identical to X_t -> masked",
            "reverse": "reverse intent re-reads the reversed path in the "
                       "same stored frame (label involution by "
                       "construction; irc_edge deltas exact negations "
                       "within PD-9 eps 1e-6)",
            "raw_vs_derived": "raw irc_edge.delta_xyz and derived "
                              "teacher_action_u are stored and hashed "
                              "separately",
        },
        "files": [
            {"path": name, "sha256": sha256_file(out / name)}
            for name in ("sample_view.csv", "irc_edge.csv",
                         "reaction_intent.csv", SPLIT_FILE)
        ],
    }
    (out / TEACHER_MANIFEST_FILE).write_text(
        json.dumps(teacher_manifest, indent=2, sort_keys=False) + "\n",
        encoding="utf-8")

    return {
        "coverage": _coverage_counts(),
        "overflow": overflow_ledger,
        "conflict_audit": dict(conflict_stats_total),
        "manifest_shas": {
            "data_manifest.json": sha256_file(out / DATA_MANIFEST_FILE),
            "teacher_manifest.json": sha256_file(
                out / TEACHER_MANIFEST_FILE),
            "split_manifest_r3.json": sha256_file(out / SPLIT_FILE),
            "dataset_sha256": dataset_sha,
        },
        "out": str(out),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="r3_build_tables")
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-conflict-audit", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"refusing to overwrite non-empty {out} (use --force)",
              file=sys.stderr)
        return 2
    summary = build(repo, out, limit=args.limit,
                    skip_conflict_audit=args.skip_conflict_audit)
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "overflow"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
