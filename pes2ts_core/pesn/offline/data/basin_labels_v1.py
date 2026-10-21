"""V5.2-R D2 phase 2 — P5 chemical-endpoint basin labels (numpy-only).

Schema ``pes2ts_v52r_basin_labels_v1`` — implements plan
``PES2TS_V52R_输入充分性与化学终点识别优化方案_v1`` §4.6 supervision as pinned
verbatim in ``outputs/pesn/v52r/d2/p5_spec.json`` (pre-registered BEFORE any
label computation or xtb call):

**B_target per curve-intent** (pre-registered construction order):

* (a) IRC reference endpoint frame — ``dataset.frame_xyz(curve,
  target_endpoint_node)``;
* (b) its graph-automorphism images — handled inside
  :func:`graph_symmetry.min_sym_rmsd` via the K-perms of the TARGET-endpoint
  graph (forward → ``g_tgt``, reverse → ``g_src``);
* (c) xtb-optimized conformer enrichment (built by the D2 phase-2 driver in
  ``scripts/agent_ops/data_gate/v52r_d2_p5_probe.py`` — subprocess is
  sanctioned there for this QC only; this module only GENERATES and
  clash-filters candidate geometries, pure numpy);
* (d) no independent IRC endpoints in this dataset (skipped).

``d_basin(X) = min_{Y ∈ B_target} min_π RMSD_K(X, π(Y))`` with the frozen
proper-rotation Kabsch (``kabsch_rmsd_v52``, imported by graph_symmetry).

**Basin label** (three-state, parallel to the d_end windows 0.10 / 0.30 Å;
literal pinned inequalities — the measure-zero ``d_basin == 0.30`` case maps
to Mask, the only deviation from d_end's inclusive-edge convention):

* ``1``    iff ``d_basin ≤ 0.10`` AND bond-edit complete;
* ``0``    iff ``d_basin > 0.30`` OR (``d_basin ≤ 0.10`` AND edit incomplete);
* ``−1``   iff ``0.10 < d_basin ≤ 0.30`` (Mask).

**Bond-edit complete** (D0-verbatim topology proxy, contract §5.3 rule 5):
EVERY intent CV-pair distance within ``EDIT_TOLERANCE_ANGSTROM`` (0.25) of the
target-endpoint frame — fixed correspondence, no automorphism.  The 22 D0.1
class-T rows therefore receive basin label 0 (never positive, never Mask:
their ``d_end ≤ 0.10`` and ``d_basin ≤ d_end``).

Conformer candidate construction (pre-registered seeds 20261021 / 20261105):
random torsion kicks (sign × Uniform(30°,150°)) on rotatable single bonds NOT
touching changing-bond atoms (both endpoints graph-degree ≥ 2), then a
Gaussian cart jitter σ=0.05 Å on H atoms, then a hard clash rejection
(any interatomic distance < 0.8 Å).  The xtb constrained optimization and the
energy-window acceptance live in the script driver (subprocess).

Pure numpy + graph_symmetry + label-window imports only — NO rdkit/sklearn.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Iterable, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.offline.data.graph_symmetry import (
    MolecularGraph,
    k_automorphism_permutations,
    min_sym_rmsd,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import KeyAtomDatasetV3
from pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1 import (
    ENDPOINT_STOP_NEGATIVE_WINDOW_ANGSTROM,
    ENDPOINT_STOP_POSITIVE_WINDOW_ANGSTROM,
    CurveEndpointStopSummary,
    EndpointStopLabelTable,
    EndpointStopSampleLabel,
)

SCHEMA: Final[str] = "pes2ts_v52r_basin_labels_v1"

#: Three-state windows — identical to the frozen d_end windows (plan §4.6).
BASIN_POSITIVE_WINDOW_ANGSTROM: Final[float] = ENDPOINT_STOP_POSITIVE_WINDOW_ANGSTROM
BASIN_NEGATIVE_WINDOW_ANGSTROM: Final[float] = ENDPOINT_STOP_NEGATIVE_WINDOW_ANGSTROM
#: D0 topology proxy tolerance (D0/D0.1 ``topology_dev_tol_angstrom``).
EDIT_TOLERANCE_ANGSTROM: Final[float] = 0.25
#: Conformer clash rejection floor (any pair below → candidate discarded).
CLASH_MIN_DISTANCE_ANGSTROM: Final[float] = 0.8
#: Gaussian cart jitter on H atoms (Å).
H_JITTER_SIGMA_ANGSTROM: Final[float] = 0.05
#: Uniform torsion-kick magnitude band (degrees), random sign.
TORSION_KICK_DEG: Final[tuple[float, float]] = (30.0, 150.0)
#: Pre-registered conformer-construction seeds (p5_spec.json).
CONFORMER_SEED_PRIMARY: Final[int] = 20261021
CONFORMER_SEED_FLIP: Final[int] = 20261105
#: Per curve-intent enrichment caps (p5_spec.json).
MAX_CONFORMERS_ACCEPTED: Final[int] = 3
MAX_GENERATION_ATTEMPTS: Final[int] = 5


# ---------------------------------------------------------------------------
# bond-edit gate (D0-verbatim topology proxy)
# ---------------------------------------------------------------------------


def cv_pair_deviations(frame_xyz: np.ndarray,
                       target_frame: np.ndarray,
                       cv_pairs: Iterable[Sequence[int]]) -> np.ndarray:
    """|d_current(i,j) − d_target(i,j)| per CV pair — fixed correspondence."""
    cur = np.asarray(frame_xyz, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(target_frame, dtype=np.float64).reshape(-1, 3)
    if cur.shape != tgt.shape:
        raise ValueError(f"frame {cur.shape} vs target {tgt.shape}")
    out: list[float] = []
    for a, b in cv_pairs:
        i, j = int(a), int(b)
        out.append(abs(float(np.linalg.norm(cur[i] - cur[j]))
                       - float(np.linalg.norm(tgt[i] - tgt[j]))))
    return np.asarray(out, dtype=np.float64)


def edit_gate_complete(frame_xyz: np.ndarray, target_frame: np.ndarray,
                       cv_pairs: Iterable[Sequence[int]], *,
                       tol: float = EDIT_TOLERANCE_ANGSTROM) -> bool:
    """True iff EVERY CV pair is within ``tol`` of the target endpoint frame.

    D0/D0.1 topology proxy semantics: deviations STRICTLY ABOVE 0.25 Å mean
    "target bond-edit incomplete" (class T / contract §5.3 rule 5), so the
    gate closes at ``> tol`` and stays open at ``≤ tol``.  An intent without
    CV pairs is vacuously complete (no edit evidence against it).
    """
    devs = cv_pair_deviations(frame_xyz, target_frame, cv_pairs)
    if devs.size == 0:
        return True
    return bool(np.all(devs <= float(tol) + 1e-12))


def basin_label_from_distance(d_basin_value: float, edit_complete: bool) -> int:
    """Frozen three-state basin label (pinned inequalities, see module doc)."""
    d = float(d_basin_value)
    if d <= BASIN_POSITIVE_WINDOW_ANGSTROM:
        return 1 if edit_complete else 0
    if d > BASIN_NEGATIVE_WINDOW_ANGSTROM:
        return 0
    return -1


# ---------------------------------------------------------------------------
# B_target
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class BasinTarget:
    """One curve-intent's B_target: reference (+ conformers) × K-perms."""

    curve_id: str
    direction: str
    target_endpoint_node: int
    k_rows: tuple[int, ...]
    ref_frame: np.ndarray                       # [N, 3] reference endpoint
    frames: tuple[np.ndarray, ...]              # ref_frame first
    provenance: tuple[str, ...]                 # "reference" | "conformer_*"
    perms: tuple[tuple[int, ...], ...]          # K-perms (identity included)

    @property
    def n_conformers(self) -> int:
        return max(0, len(self.frames) - 1)


def endpoint_graph(intent, direction: str) -> MolecularGraph:
    """Target-endpoint graph of an intent (forward → g_tgt, reverse → g_src)."""
    if direction == "forward":
        bonds = intent.g_tgt_bonds
    elif direction == "reverse":
        bonds = intent.g_src_bonds
    else:
        raise ValueError(f"direction must be forward|reverse, got {direction!r}")
    if not intent.g_atoms:
        raise ValueError(f"intent {intent.intent_id} has no g_atoms payload")
    return MolecularGraph.from_intent_payload(intent.g_atoms, bonds)


def build_reference_targets(
    dataset: KeyAtomDatasetV3,
    d_end_table: EndpointStopLabelTable,
) -> dict[tuple[str, str], BasinTarget]:
    """Reference-only B_target for EVERY curve-intent (all 50 curves, QC use)."""
    targets: dict[tuple[str, str], BasinTarget] = {}
    for (curve_id, _intent_id), intent in sorted(dataset.intents.items()):
        direction = intent.direction
        summary = d_end_table.curves.get(curve_id)
        if summary is None:
            raise KeyError(f"no curve summary for {curve_id}")
        target_node = (int(summary.endpoint_b_node) if direction == "forward"
                       else int(summary.endpoint_a_node))
        ref = np.asarray(dataset.frame_xyz(curve_id, target_node),
                         dtype=np.float64)
        k_rows = tuple(int(k) for k in intent.key_rows())
        graph = endpoint_graph(intent, direction)
        perms = k_automorphism_permutations(graph, k_rows)
        targets[(curve_id, direction)] = BasinTarget(
            curve_id=curve_id, direction=direction,
            target_endpoint_node=target_node, k_rows=k_rows,
            ref_frame=ref, frames=(ref,), provenance=("reference",),
            perms=perms)
    return targets


def with_conformers(target: BasinTarget,
                    conformers: Sequence[np.ndarray]) -> BasinTarget:
    """Return a copy of ``target`` with xtb-optimized conformers appended."""
    extra = tuple(np.asarray(c, dtype=np.float64) for c in conformers)
    return BasinTarget(
        curve_id=target.curve_id, direction=target.direction,
        target_endpoint_node=target.target_endpoint_node,
        k_rows=target.k_rows, ref_frame=target.ref_frame,
        frames=target.frames + extra,
        provenance=target.provenance + tuple(
            f"conformer_{i + 1}" for i in range(len(extra))),
        perms=target.perms)


def d_basin(x_frame: np.ndarray, target: BasinTarget) -> float:
    """``min_{Y ∈ B_target} min_π RMSD_K(X, π(Y))`` (proper-rotation Kabsch)."""
    x = np.asarray(x_frame, dtype=np.float64).reshape(-1, 3)
    xk = x[list(target.k_rows)]
    best = float("inf")
    for y in target.frames:
        val = min_sym_rmsd(xk, np.asarray(y, dtype=np.float64)[list(target.k_rows)],
                           target.perms)
        if val < best:
            best = val
    return float(best)


# ---------------------------------------------------------------------------
# conformer candidate construction (pure numpy; xtb lives in the script)
# ---------------------------------------------------------------------------


def changing_bond_atoms(intent) -> frozenset[int]:
    """Atoms of every pair whose bond order changes (b_src ≠ b_tgt)."""
    out: set[int] = set()
    for (i, j), (b_src, b_tgt) in intent.pair_orders().items():
        if b_src != b_tgt:
            out.add(int(i))
            out.add(int(j))
    return frozenset(out)


def rotatable_bonds(graph: MolecularGraph,
                    excluded_atoms: Iterable[int],
                    ) -> tuple[tuple[int, int], ...]:
    """Single-order bonds not touching ``excluded_atoms``, degree ≥ 2 both ends."""
    excluded = {int(a) for a in excluded_atoms}
    degree: dict[int, int] = {}
    for i, j, _o in graph.bonds:
        degree[i] = degree.get(i, 0) + 1
        degree[j] = degree.get(j, 0) + 1
    out: list[tuple[int, int]] = []
    for i, j, order in graph.bonds:
        if round(float(order), 6) != 1.0:
            continue
        if i in excluded or j in excluded:
            continue
        if degree.get(i, 0) < 2 or degree.get(j, 0) < 2:
            continue
        out.append((int(i), int(j)))
    return tuple(sorted(out))


def _components_without_edge(graph: MolecularGraph, cut: tuple[int, int]
                             ) -> frozenset[int]:
    """Atom set of the component containing the bond's SECOND atom once the
    edge is removed (BFS that refuses to cross the cut edge)."""
    a, b = cut
    adj: dict[int, list[int]] = {}
    for i, j, _o in graph.bonds:
        if {i, j} == {a, b}:
            continue
        adj.setdefault(i, []).append(j)
        adj.setdefault(j, []).append(i)
    seen = {b}
    stack = [b]
    while stack:
        v = stack.pop()
        for u in adj.get(v, ()):
            if u not in seen:
                seen.add(u)
                stack.append(u)
    return frozenset(seen)


def _rotate_about_axis(points: np.ndarray, origin: np.ndarray,
                       axis: np.ndarray, theta: float) -> np.ndarray:
    """Rodrigues rotation of ``points`` about the line (origin, axis unit)."""
    u = axis / np.linalg.norm(axis)
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    rel = points - origin
    cross = np.cross(u, rel)
    dot = rel @ u
    return origin + rel * cos_t + np.cross(u, rel) * sin_t + u * dot[:, None] * (1.0 - cos_t)


def generate_conformer(ref_frame: np.ndarray,
                       bonds: Sequence[tuple[int, int]],
                       graph: MolecularGraph,
                       atomic_numbers: Sequence[int],
                       rng: np.random.RandomState) -> np.ndarray:
    """One pre-registered candidate: torsion kicks on every rotatable bond
    (θ = sign × Uniform(30°,150°), rotating the SMALLER side; ties → second
    atom's side) then H-only Gaussian jitter σ=0.05 Å."""
    ref = np.asarray(ref_frame, dtype=np.float64).reshape(-1, 3)
    coords = ref.copy()
    n = coords.shape[0]
    for (i, j) in bonds:
        side_j = _components_without_edge(graph, (i, j))
        side_i = frozenset(set(range(n)) - set(side_j))
        side = side_j if len(side_j) <= len(side_i) else side_i
        if not side:
            continue
        sign = -1.0 if rng.rand() < 0.5 else 1.0
        theta = np.deg2rad(sign * rng.uniform(*TORSION_KICK_DEG))
        axis = ref[j] - ref[i]
        if float(np.linalg.norm(axis)) < 1e-12:
            continue
        idx = np.asarray(sorted(side), dtype=np.int64)
        coords[idx] = _rotate_about_axis(coords[idx], ref[i], axis, theta)
    z = np.asarray(tuple(int(a) for a in atomic_numbers), dtype=np.int64)
    h_mask = z == 1
    if bool(h_mask.any()):
        coords[h_mask] += rng.normal(0.0, H_JITTER_SIGMA_ANGSTROM,
                                     size=(int(h_mask.sum()), 3))
    return coords


def has_clash(frame: np.ndarray,
              min_dist: float = CLASH_MIN_DISTANCE_ANGSTROM) -> bool:
    """True when ANY interatomic distance (incl. 1-2 bonds) is < ``min_dist``."""
    pts = np.asarray(frame, dtype=np.float64).reshape(-1, 3)
    diff = pts[:, None, :] - pts[None, :, :]
    dist = np.sqrt((diff * diff).sum(axis=-1))
    np.fill_diagonal(dist, np.inf)
    return bool(dist.min() < float(min_dist))


# ---------------------------------------------------------------------------
# basin label table
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BasinSampleLabel:
    """One sample row's P5 basin label (schema v1)."""

    sample_id: str
    curve_id: str
    node_index: int
    intent_id: str
    direction: str                       # "forward" | "reverse"
    target_endpoint: str                 # "A" (reverse) | "B" (forward)
    target_endpoint_node: int
    d_basin: float
    edit_complete: bool                  # D0-verbatim CV proxy (≤ 0.25 Å)
    basin_label: int                     # 1 / 0 / −1 (Mask)
    degenerate_flag: bool                # contract §5.3 rule 4 (same as d_end)
    move_status: str


@dataclass(frozen=True)
class BasinLabelTable:
    """Basin labels over a FULL dataset load (dev + sealed, QC counting)."""

    by_sample: Mapping[str, BasinSampleLabel]
    curves: Mapping[str, CurveEndpointStopSummary]
    positive_window: float
    negative_window: float

    @property
    def n_rows(self) -> int:
        return len(self.by_sample)

    def count_label(self, label: int) -> int:
        return sum(1 for lab in self.by_sample.values()
                   if lab.basin_label == label)

    @property
    def n_positive(self) -> int:
        return self.count_label(1)

    @property
    def n_negative(self) -> int:
        return self.count_label(0)

    @property
    def n_mask(self) -> int:
        return self.count_label(-1)

    def labels_by_sample(self) -> dict[str, int]:
        return {sid: lab.basin_label for sid, lab in self.by_sample.items()}

    def as_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "positive_window_angstrom": self.positive_window,
            "negative_window_angstrom": self.negative_window,
            "n_rows": self.n_rows,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "n_mask": self.n_mask,
            "curves": [self.curves[cid].__dict__ for cid in sorted(self.curves)],
            "rows": [
                {
                    "sample_id": lab.sample_id,
                    "curve_id": lab.curve_id,
                    "node_index": lab.node_index,
                    "intent_id": lab.intent_id,
                    "direction": lab.direction,
                    "target_endpoint": lab.target_endpoint,
                    "target_endpoint_node": lab.target_endpoint_node,
                    "d_basin": lab.d_basin,
                    "edit_complete": lab.edit_complete,
                    "basin_label": lab.basin_label,
                    "degenerate_flag": lab.degenerate_flag,
                    "move_status": lab.move_status,
                }
                for sid, lab in sorted(self.by_sample.items())],
        }


def build_basin_labels(
    dataset: KeyAtomDatasetV3,
    d_end_table: EndpointStopLabelTable,
    targets: Mapping[tuple[str, str], BasinTarget],
) -> BasinLabelTable:
    """Basin labels for EVERY loaded row (dev + sealed — label-QC counting)."""
    labels: dict[str, BasinSampleLabel] = {}
    curve_rows: dict[str, list[BasinSampleLabel]] = {}
    cache: dict[tuple[str, str, int], tuple[float, bool]] = {}
    for rec in dataset.samples:
        cid = rec.curve_id
        direction = rec.intent.direction
        target = targets[(cid, direction)]
        ck = (cid, direction, int(rec.node_index))
        cached = cache.get(ck)
        frame = np.asarray(dataset.frame_xyz(cid, rec.node_index),
                           dtype=np.float64)
        if cached is None:
            db = d_basin(frame, target)
            complete = edit_gate_complete(frame, target.ref_frame,
                                          rec.intent.cv_coordinates)
            cache[ck] = (db, complete)
        else:
            db, complete = cached
        lab_end = d_end_table.by_sample[rec.sample_id]
        lab = BasinSampleLabel(
            sample_id=rec.sample_id, curve_id=cid,
            node_index=int(rec.node_index), intent_id=rec.intent.intent_id,
            direction=direction, target_endpoint=lab_end.target_endpoint,
            target_endpoint_node=int(lab_end.target_endpoint_node),
            d_basin=float(db), edit_complete=bool(complete),
            basin_label=basin_label_from_distance(db, complete),
            degenerate_flag=bool(lab_end.degenerate_flag),
            move_status=lab_end.move_status)
        labels[rec.sample_id] = lab
        curve_rows.setdefault(cid, []).append(lab)

    import statistics

    summaries: dict[str, CurveEndpointStopSummary] = {}
    for cid, rows in sorted(curve_rows.items()):
        base = d_end_table.curves[cid]
        db = sorted(r.d_basin for r in rows)
        summaries[cid] = CurveEndpointStopSummary(
            curve_id=cid, endpoint_a_node=base.endpoint_a_node,
            endpoint_b_node=base.endpoint_b_node, ts_index=base.ts_index,
            n_rows=len(rows),
            n_pos=sum(1 for r in rows if r.basin_label == 1),
            n_neg=sum(1 for r in rows if r.basin_label == 0),
            n_mask=sum(1 for r in rows if r.basin_label == -1),
            n_degenerate=sum(1 for r in rows if r.degenerate_flag),
            d_end_min=float(db[0]),
            d_end_median=float(statistics.median(db)),
            d_end_max=float(db[-1]))
    return BasinLabelTable(
        by_sample=labels, curves=summaries,
        positive_window=BASIN_POSITIVE_WINDOW_ANGSTROM,
        negative_window=BASIN_NEGATIVE_WINDOW_ANGSTROM)


# ---------------------------------------------------------------------------
# adapters + QC accounting (probe reuse / validity checks)
# ---------------------------------------------------------------------------


def as_endpoint_stop_labels(basin_table: BasinLabelTable) -> EndpointStopLabelTable:
    """Basin table reshaped into the probe runner's label-table interface
    (``endpoint_stop_label`` carries the BASIN label, ``d_end_angstrom`` carries
    ``d_basin``) so the frozen D1/D2 probe machinery runs unchanged."""
    by_sample = {
        sid: EndpointStopSampleLabel(
            sample_id=lab.sample_id, curve_id=lab.curve_id,
            node_index=lab.node_index, intent_id=lab.intent_id,
            direction=lab.direction, target_endpoint=lab.target_endpoint,
            target_endpoint_node=lab.target_endpoint_node,
            d_end_angstrom=float(lab.d_basin),
            endpoint_stop_label=int(lab.basin_label),
            degenerate_flag=bool(lab.degenerate_flag),
            move_status=lab.move_status)
        for sid, lab in basin_table.by_sample.items()}
    return EndpointStopLabelTable(
        by_sample=by_sample, curves=basin_table.curves,
        positive_window=basin_table.positive_window,
        negative_window=basin_table.negative_window)


def restrict_to(table: EndpointStopLabelTable,
                keep: Iterable[str]) -> EndpointStopLabelTable:
    """Mask every row OUTSIDE ``keep`` to label −1 (same rows, one label set).

    Used to build the INTERSECTION population: both schemes evaluate exactly
    the same rows while keeping their own label values.
    """
    keep_set = frozenset(str(s) for s in keep)
    by_sample = {
        sid: (lab if sid in keep_set else EndpointStopSampleLabel(
            sample_id=lab.sample_id, curve_id=lab.curve_id,
            node_index=lab.node_index, intent_id=lab.intent_id,
            direction=lab.direction, target_endpoint=lab.target_endpoint,
            target_endpoint_node=lab.target_endpoint_node,
            d_end_angstrom=lab.d_end_angstrom, endpoint_stop_label=-1,
            degenerate_flag=lab.degenerate_flag, move_status=lab.move_status))
        for sid, lab in table.by_sample.items()}
    curve_rows: dict[str, list[EndpointStopSampleLabel]] = {}
    for lab in by_sample.values():
        curve_rows.setdefault(lab.curve_id, []).append(lab)
    import statistics

    summaries: dict[str, CurveEndpointStopSummary] = {}
    for cid, rows in sorted(curve_rows.items()):
        base = table.curves[cid]
        dvals = sorted(r.d_end_angstrom for r in rows)
        summaries[cid] = CurveEndpointStopSummary(
            curve_id=cid, endpoint_a_node=base.endpoint_a_node,
            endpoint_b_node=base.endpoint_b_node, ts_index=base.ts_index,
            n_rows=len(rows),
            n_pos=sum(1 for r in rows if r.endpoint_stop_label == 1),
            n_neg=sum(1 for r in rows if r.endpoint_stop_label == 0),
            n_mask=sum(1 for r in rows if r.endpoint_stop_label == -1),
            n_degenerate=sum(1 for r in rows if r.degenerate_flag),
            d_end_min=float(dvals[0]) if dvals else 0.0,
            d_end_median=float(statistics.median(dvals)) if dvals else 0.0,
            d_end_max=float(dvals[-1]) if dvals else 0.0)
    return EndpointStopLabelTable(by_sample=by_sample, curves=summaries,
                                  positive_window=table.positive_window,
                                  negative_window=table.negative_window)


def check_t_rows(sample_ids: Iterable[str],
                 basin_table: BasinLabelTable) -> dict:
    """Validity (3): every D0.1 class-T row must carry basin label 0."""
    rows: list[dict] = []
    missing: list[str] = []
    for sid in sample_ids:
        lab = basin_table.by_sample.get(sid)
        if lab is None:
            missing.append(sid)
            continue
        rows.append({
            "sample_id": sid, "curve_id": lab.curve_id,
            "basin_label": int(lab.basin_label),
            "d_basin": round(float(lab.d_basin), 6),
            "edit_complete": bool(lab.edit_complete),
            "ok": lab.basin_label == 0,
        })
    n_ok = sum(1 for r in rows if r["ok"])
    return {"n_rows": len(rows), "n_positive": sum(
                1 for r in rows if r["basin_label"] == 1),
            "n_mask": sum(1 for r in rows if r["basin_label"] == -1),
            "n_negative": sum(1 for r in rows if r["basin_label"] == 0),
            "n_ok": n_ok, "missing": missing, "pass": (
                not missing and n_ok == len(sample_ids)),
            "rows": rows}


def flip_stats(labels_a: Mapping[str, int], labels_b: Mapping[str, int],
               sample_ids: Iterable[str]) -> dict:
    """Validity (4): three-state label flips between two enrichment seeds."""
    ids = [str(s) for s in sample_ids]
    transitions: dict[str, int] = {}
    n_flips = 0
    for sid in ids:
        a = labels_a.get(sid)
        b = labels_b.get(sid)
        if a is None or b is None:
            continue
        if a != b:
            n_flips += 1
            key = f"{a}->{b}"
            transitions[key] = transitions.get(key, 0) + 1
    n = len(ids)
    return {"n_rows": n, "n_flips": n_flips,
            "flip_rate": (n_flips / n) if n else None,
            "transitions": dict(sorted(transitions.items()))}


def population_counts(basin_table: BasinLabelTable,
                      sample_ids: Iterable[str]) -> dict:
    """pos/neg/mask (+ usable) for a row subset (dev / full table)."""
    ids = [str(s) for s in sample_ids]
    pos = neg = mask = deg = 0
    usable_pos = usable_neg = 0
    for sid in ids:
        lab = basin_table.by_sample.get(sid)
        if lab is None:
            continue
        if lab.degenerate_flag:
            deg += 1
        if lab.basin_label == 1:
            pos += 1
        elif lab.basin_label == 0:
            neg += 1
        else:
            mask += 1
        if not lab.degenerate_flag and lab.basin_label >= 0:
            if lab.basin_label == 1:
                usable_pos += 1
            else:
                usable_neg += 1
    usable = usable_pos + usable_neg
    return {
        "n_rows": len(ids), "positive": pos, "negative": neg, "mask": mask,
        "degenerate_rows": deg,
        "usable_rows": usable, "usable_positive": usable_pos,
        "usable_negative": usable_neg,
        "usable_positive_rate": (usable_pos / usable) if usable else None,
    }


__all__ = [
    "SCHEMA",
    "BASIN_POSITIVE_WINDOW_ANGSTROM",
    "BASIN_NEGATIVE_WINDOW_ANGSTROM",
    "EDIT_TOLERANCE_ANGSTROM",
    "CLASH_MIN_DISTANCE_ANGSTROM",
    "H_JITTER_SIGMA_ANGSTROM",
    "TORSION_KICK_DEG",
    "CONFORMER_SEED_PRIMARY",
    "CONFORMER_SEED_FLIP",
    "MAX_CONFORMERS_ACCEPTED",
    "MAX_GENERATION_ATTEMPTS",
    "cv_pair_deviations",
    "edit_gate_complete",
    "basin_label_from_distance",
    "BasinTarget",
    "endpoint_graph",
    "build_reference_targets",
    "with_conformers",
    "d_basin",
    "changing_bond_atoms",
    "rotatable_bonds",
    "generate_conformer",
    "has_clash",
    "BasinSampleLabel",
    "BasinLabelTable",
    "build_basin_labels",
    "as_endpoint_stop_labels",
    "restrict_to",
    "check_t_rows",
    "flip_stats",
    "population_counts",
]
