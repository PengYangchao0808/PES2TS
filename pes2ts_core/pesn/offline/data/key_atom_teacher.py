"""V5 key-atom teacher: reference-gauge Kabsch, tail marking, contact solver.

Implements ADR-0026 §2.2 (frame contract) / §2.3 (teacher labels) / §2.5
(conflict audit) and sprint plan §1.1–§1.3:

* reference block S per branch from the START endpoint covalent graph:
  non-key heavy atoms within 1–2 covalent hops of the key set, restricted to
  the rigid block(s) obtained by cutting non-ring single bonds whose both
  endpoints have degree >= 2 (terminal degree-1 tips stay attached to their
  kept neighbor); weights ``w_i = [i in S] / (1 + hop(i,A))^2``, W
  normalized; a scatter gate (lambda1 > 0.01 A^2, lambda2/lambda1 > 1e-3)
  else the branch is exported ``gauge_unresolved`` — never a global axis;
* weighted Kabsch in float64 with the ADR row-vector convention
  ``C=(Y-muY)^T W (X-muX)=U S V^T``, ``R=U diag(1,1,det(UV^T)) V^T``
  (proper rotation only — reflections are never allowed to "fix" enantiomers);
* flat-tail marking per plan §1.1 (vq/f formulas, 20% key-path cap, TS end
  never cut) with training weight q=0.25 inside the tail;
* the first-contact teacher: piecewise-linear segments over future frames
  aligned label-side into X_t with the FIXED branch S/W; per-segment
  quadratic root find for RMS_A=d0 (root tol 1e-8) and per-atom cap=0.20 A,
  earliest event along the path wins; TS-terminal and no-future-step
  semantics per ADR §2.3;
* the bounded fit50-internal conflict audit (plan §1.2) with automorphism
  orbit enumeration capped at 128 (``mapping_group_overflow`` beyond);
* the frame-contract writer and the stage-0 ``prepare_audit`` orchestrator
  (guarded CLI; refuses to run without the frozen V5 split manifest).

NO energy/gradient value is ever read into or written from this module's
outputs: only coordinates cross the teacher boundary.  All geometry is pure
numpy float64; float32 appears only at shard-write time.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.contracts.key_atom import (
    CONFLICT_AUDIT_SCHEMA,
    FRAME_CONTRACT_SCHEMA,
    KeyAtomTask,
    REASON_GAUGE_UNRESOLVED,
    REASON_NO_FUTURE_STEP,
    REASON_OK,
    REASON_REFERENCE_INVALID,
    SPLIT_SCHEMA as V5_SPLIT_SCHEMA,
    build_key_atom_task,
)
from pes2ts_core.pesn.offline.data.dual_ts_preprocess import (
    DualTSPreprocessError,
    order_frames_and_energy,
    split_dual_branches,
)
from pes2ts_core.utils.hashing import sha256_file
from pes2ts_core.utils.jsonio import read_json, write_json

NATIVE_SCHEMA: Final[str] = "pesn_v5_native_v1"
TEACHER_PROGRESS_SCHEMA: Final[str] = "pesn_v5_prepare_audit_progress_v1"

#: teacher spans (plan §5.2: same label machinery, three requested distances)
D0_VALUES: Final[tuple[float, ...]] = (0.05, 0.08, 0.12)
MAIN_D0: Final[float] = 0.08

#: per-atom teacher cap (plan §1.3 / ADR §2.3)
TEACHER_CAP_ANGSTROM: Final[float] = 0.20

#: flat-tail constants (plan §1.1)
TAIL_F_THRESHOLD: Final[float] = 0.05
TAIL_MIN_INTERVALS: Final[int] = 5
TAIL_Q_RANGE_MAX: Final[float] = 0.02
TAIL_Q_WEIGHT: Final[float] = 0.25
TAIL_MAX_FRACTION: Final[float] = 0.20
TAIL_CONFIRM_CONSECUTIVE_HIGH: Final[int] = 3

#: reference scatter gate (ADR §2.2)
GAUGE_E1_MIN_ANGSTROM2: Final[float] = 0.01
GAUGE_E2_RATIO_MIN: Final[float] = 1e-3

#: contact root tolerance (ADR §2.3)
ROOT_TOL: Final[float] = 1e-8
#: "no future nonzero step" identity tolerance
FUTURE_STEP_TOL: Final[float] = 1e-9

#: conflict-audit thresholds (plan §1.2)
CONFLICT_FULL_RMSD: Final[float] = 0.03
CONFLICT_KEY_RMSD: Final[float] = 0.02
CONFLICT_LABEL_RMS: Final[float] = 0.02
CONFLICT_COSINE_NEGATIVE: Final[float] = 0.0
NEAR_CONFLICT_COSINE: Final[float] = 0.25
NEAR_CONFLICT_WEIGHT: Final[float] = 0.25
ZERO_WEIGHT: Final[float] = 0.0
ORBIT_ENUM_CAP: Final[int] = 128


class KeyAtomTeacherError(RuntimeError):
    """Typed teacher-build refusal (never a silent drop)."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# weighted Kabsch (ADR-0026 §2.2; plan §1.3) — float64, proper rotations only
# ---------------------------------------------------------------------------


def weighted_kabsch(
    Y: np.ndarray, X: np.ndarray, w: np.ndarray,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Align row-vector point set Y onto X under diagonal weights w.

    Implements the ADR convention verbatim:

    ``C = (Y-muY)^T W (X-muX) = U S V^T``; ``R = U diag(1,1,det(UV^T)) V^T``;
    ``aligned = (Y-muY) R + muX`` with W = diag(w/sum(w)), centroids taken
    under W.  R is always a proper rotation (det = +1); reflections are never
    used, so genuine enantiomeric differences survive the registration.

    Returns ``(aligned_rows, (R, muY, muX))``; apply the same rigid transform
    to full frames via :func:`apply_rigid_transform`.
    """
    y = np.asarray(Y, dtype=np.float64)
    x = np.asarray(X, dtype=np.float64)
    weights = np.asarray(w, dtype=np.float64)
    if y.shape != x.shape or y.ndim != 2 or y.shape[1] != 3:
        raise KeyAtomTeacherError(
            "KABSCH_SHAPE", f"incompatible shapes {y.shape} vs {x.shape}")
    if weights.shape != (y.shape[0],):
        raise KeyAtomTeacherError(
            "KABSCH_WEIGHTS", f"weights shape {weights.shape} != ({y.shape[0]},)")
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise KeyAtomTeacherError("KABSCH_WEIGHTS", "weight sum must be > 0")
    wn = weights / total
    mu_y = wn @ y
    mu_x = wn @ x
    a = y - mu_y
    b = x - mu_x
    c = a.T @ (wn[:, None] * b)
    u, _s, vt = np.linalg.svd(c)
    corr = np.array([1.0, 1.0, 1.0 if np.linalg.det(u @ vt) >= 0.0 else -1.0])
    rotation = (u * corr) @ vt  # U diag(1,1,det(UV^T)) V^T
    aligned = a @ rotation + mu_x
    return aligned, (rotation, mu_y, mu_x)


def apply_rigid_transform(
    X: np.ndarray, transform: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Apply a ``(R, muY, muX)`` transform from :func:`weighted_kabsch`."""
    rotation, mu_y, mu_x = transform
    return (np.asarray(X, dtype=np.float64) - mu_y) @ rotation + mu_x


def kabsch_rmsd(Y: np.ndarray, X: np.ndarray) -> float:
    """Equal-weight proper-rotation RMSD between two point sets."""
    aligned, _ = weighted_kabsch(Y, X, np.ones(len(Y), dtype=np.float64))
    return float(np.sqrt(np.mean(np.sum((aligned - X) ** 2, axis=1))))


# ---------------------------------------------------------------------------
# reference block S (plan §1.3; simplified rigid-block rule, documented)
# ---------------------------------------------------------------------------


def _adjacency(n: int, bonds: Sequence[tuple[int, int, float]]):
    adj: list[set[int]] = [set() for _ in range(n)]
    for i, j, _order in bonds:
        adj[int(i)].add(int(j))
        adj[int(j)].add(int(i))
    return adj


def _connected(adj: Sequence[set[int]], src: int, dst: int,
               banned_edge: tuple[int, int] | None = None) -> bool:
    seen = {src}
    stack = [src]
    while stack:
        node = stack.pop()
        if node == dst:
            return True
        for nb in adj[node]:
            if banned_edge is not None and (
                    {node, nb} == set(banned_edge)):
                continue
            if nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return dst in seen


def _ring_bonds(n: int, bonds: Sequence[tuple[int, int, float]]):
    """Bond pairs that lie on a cycle (edge removal + connectivity test)."""
    adj = _adjacency(n, bonds)
    rings = set()
    for i, j, _order in bonds:
        pair = (min(int(i), int(j)), max(int(i), int(j)))
        if pair in rings:
            continue
        if _connected(adj, int(i), int(j), banned_edge=pair):
            rings.add(pair)
    return rings


def compute_reference_block(
    atomic_numbers: Sequence[int],
    start_bonds: Sequence[tuple[int, int, float]],
    key_rows: Sequence[int],
) -> dict:
    """Reference block rows/weights/hops for one branch (plan §1.3).

    Simplified, documented rigid-block rule: cut every NON-ring SINGLE bond
    whose endpoints both have covalent degree >= 2 (a free rotor); the
    connected components of the remaining graph are the rigid blocks.
    Terminal degree-1 tips (CH3/CF3 hydrogens etc.) are attached through
    bonds with a degree-1 endpoint, hence never cut — they stay with their
    neighbor's block.  S then keeps the non-key HEAVY atoms with hop distance
    1 or 2 from the key set that share a rigid block with a key atom.
    """
    z = [int(v) for v in atomic_numbers]
    n = len(z)
    key = {int(r) for r in key_rows}
    bonds = [(int(i), int(j), float(o)) for i, j, o in start_bonds]
    adj = _adjacency(n, bonds)
    degrees = [len(nb) for nb in adj]
    rings = _ring_bonds(n, bonds)

    # rigid components: union-find over non-cut bonds
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    cut_count = 0
    for i, j, order in bonds:
        pair = (min(i, j), max(i, j))
        rotor = (abs(order - 1.0) < 1e-9 and pair not in rings
                 and degrees[i] >= 2 and degrees[j] >= 2)
        if rotor:
            cut_count += 1
            continue
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj
    key_roots = {find(r) for r in key}

    # hops from the key set over the FULL start graph (level-by-level BFS;
    # a LIFO stack would be a DFS and could overestimate hop distances)
    hops = {r: 0 for r in key}
    frontier = sorted(key)
    while frontier:
        next_frontier: list[int] = []
        for node in frontier:
            for nb in adj[node]:
                if nb not in hops:
                    hops[nb] = hops[node] + 1
                    next_frontier.append(nb)
        frontier = next_frontier

    rows: list[int] = []
    weights = np.zeros(n, dtype=np.float64)
    hop_of: dict[int, int] = {}
    for i in range(n):
        if i in key or z[i] <= 1:  # key atoms and hydrogens are never reference
            continue
        hop = hops.get(i)
        if hop is None or not (1 <= hop <= 2):
            continue
        if find(i) not in key_roots:  # separated from A by a free rotor
            continue
        rows.append(i)
        hop_of[i] = hop
        weights[i] = 1.0 / (1.0 + hop) ** 2

    # Tiered fallback candidates (plan §1.3 "候选块按距 A、尺寸选择"): t0 is
    # the strict rigid 1-2-hop block; larger pools follow for molecules
    # whose rigid fragments are too small/degenerate to pass the scatter
    # gate.  The FIRST tier passing reference_gate(pos_start) is used; if
    # none passes, the branch is gauge_unresolved exactly as before.
    tiers: list[dict] = []

    def _tier(name: str, predicate) -> None:
        t_rows: list[int] = []
        t_hops: dict[int, int] = {}
        t_weights = np.zeros(n, dtype=np.float64)
        for i in range(n):
            if i in key or z[i] <= 1:
                continue
            hop = hops.get(i)
            if hop is None or not predicate(hop, i):
                continue
            t_rows.append(i)
            t_hops[i] = hop
            t_weights[i] = 1.0 / (1.0 + hop) ** 2
        tiers.append({"tier": name, "rows": np.asarray(
            sorted(t_rows), dtype=np.int64), "weights": t_weights,
            "hops": t_hops})

    _tier("rigid_1_2hop", lambda hop, i: (
        1 <= hop <= 2 and find(i) in key_roots))
    _tier("any_1_2hop", lambda hop, i: 1 <= hop <= 2)
    _tier("any_3hop", lambda hop, i: 1 <= hop <= 3)
    _tier("all_nonkey_heavy", lambda hop, i: True)
    return {
        "rows": np.asarray(sorted(rows), dtype=np.int64),
        "weights": weights,
        "hops": hop_of,
        "n_rotor_bonds_cut": cut_count,
        "degrees": degrees,
        "tiers": tiers,
    }


def reference_gate(
    pos_start: np.ndarray, rows: np.ndarray, weights: np.ndarray,
) -> tuple[bool, list[float]]:
    """Scatter gate on the START frame (ADR §2.2): lambda1 > 0.01 A^2 and
    lambda2/lambda1 > 1e-3.  Planar references are fine — no third
    eigenvalue is required."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return False, [0.0, 0.0, 0.0]
    w = np.asarray(weights, dtype=np.float64)[rows]
    total = float(w.sum())
    if total <= 0.0:
        return False, [0.0, 0.0, 0.0]
    wn = w / total
    points = np.asarray(pos_start, dtype=np.float64)[rows]
    mu = wn @ points
    a = points - mu
    cov = (a * wn[:, None]).T @ a
    evals = np.linalg.eigvalsh(cov)[::-1]  # descending
    e1 = float(evals[0])
    e2 = float(evals[1]) if len(evals) > 1 else 0.0
    ok = e1 > GAUGE_E1_MIN_ANGSTROM2 and (
        (e2 / e1) > GAUGE_E2_RATIO_MIN if e1 > 0 else False)
    return bool(ok), [float(v) for v in evals]


# ---------------------------------------------------------------------------
# key-atom path arc + flat-tail marking (plan §1.1)
# ---------------------------------------------------------------------------


def key_path_arc(pos: np.ndarray, key_rows: Sequence[int]) -> np.ndarray:
    """Per-step key-atom RMS arc ``dsA[j] = RMS_A(X[j+1]-X[j])`` (length T-1)."""
    x = np.asarray(pos, dtype=np.float64)
    key = np.asarray(key_rows, dtype=np.int64)
    if key.size == 0:
        return np.zeros(max(len(x) - 1, 0), dtype=np.float64)
    diff = x[1:, key, :] - x[:-1, key, :]
    return np.sqrt(np.mean(np.sum(diff * diff, axis=2), axis=1))


def mark_flat_tail(
    pos: np.ndarray,
    key_rows: Sequence[int],
    k_pairs: Sequence[tuple[int, int, float, float, float]],
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Flat-tail marking per plan §1.1 on the NATIVE branch frames.

    Returns ``(q_frame[T] float32, tail_flag[T] bool, meta)``.  q_frame is
    0.25 inside the flat tail (training weight) and 1.0 elsewhere.  The tail
    is the maximal contiguous prefix (from the START end) with f < 0.05,
    confirmed ended by 3 consecutive f >= 0.05, requiring >= 5 native
    intervals and ``max_l range(q_l) < 0.02 A`` over the prefix; the removed
    key-path arc may not exceed 20% of the branch (exceed -> whole branch
    ``tail_review``, frames kept, weight stays 1, never auto-extended).  The
    TS end is never cut.
    """
    x = np.asarray(pos, dtype=np.float64)
    n_frames = len(x)
    key = np.asarray(key_rows, dtype=np.int64)
    meta: dict = {"tail_review": False, "tail_rows": [], "reason": "",
                  "uninformative_whole": False,
                  "uninformative_non_tail": False}
    q_frame = np.ones(n_frames, dtype=np.float32)
    tail_flag = np.zeros(n_frames, dtype=bool)
    if n_frames < 3 or key.size == 0 or not k_pairs:
        meta["reason"] = "insufficient_frames_or_keys"
        return q_frame, tail_flag, meta

    # q_l[j] per K pair
    pair_list = [(int(p[0]), int(p[1])) for p in k_pairs]
    q = np.empty((len(pair_list), n_frames), dtype=np.float64)
    for l, (i, j) in enumerate(pair_list):
        q[l] = np.linalg.norm(x[:, i, :] - x[:, j, :], axis=1)

    # sA: cumulative key-atom arc (denominator of vq is the 2-step arc)
    ds = key_path_arc(x, key)
    s_a = np.concatenate([[0.0], np.cumsum(ds)])  # length T

    # vq[j] = RMS_l(q_l[j+1]-q_l[j-1]) / max(sA[j+1]-sA[j-1], 1e-6)
    vq = np.zeros(n_frames, dtype=np.float64)
    for j in range(n_frames):
        if j == 0:
            num = np.sqrt(np.mean((q[:, 1] - q[:, 0]) ** 2))
            den = max(float(s_a[1] - s_a[0]), 1e-6)
        elif j == n_frames - 1:
            num = np.sqrt(np.mean((q[:, -1] - q[:, -2]) ** 2))
            den = max(float(s_a[-1] - s_a[-2]), 1e-6)
        else:
            num = np.sqrt(np.mean((q[:, j + 1] - q[:, j - 1]) ** 2))
            den = max(float(s_a[j + 1] - s_a[j - 1]), 1e-6)
        vq[j] = num / den
    q90 = max(float(np.quantile(vq, 0.9)), 1e-3)
    f = vq / q90

    # whole-branch uninformative flags (meta only; labels stay eligible,
    # reason stays ok, q stays 1 — plan §1.1 "keep, independent slice")
    q_range_whole = float(np.max(np.ptp(q, axis=1))) if q.size else 0.0
    if q_range_whole < TAIL_Q_RANGE_MAX:
        meta["uninformative_whole"] = True

    low = f < TAIL_F_THRESHOLD
    # maximal contiguous low-f prefix from the START end
    tail_len = 0
    while tail_len < n_frames and low[tail_len]:
        tail_len += 1
    # the scan is "confirmed" ended by 3 consecutive high f (recorded only;
    # contiguity already bounds the prefix)
    confirmed_end = False
    for j in range(tail_len, n_frames - TAIL_CONFIRM_CONSECUTIVE_HIGH + 1):
        if bool(np.all(f[j:j + TAIL_CONFIRM_CONSECUTIVE_HIGH] >= TAIL_F_THRESHOLD)):
            confirmed_end = True
            break
    meta["f_profile"] = [float(v) for v in f]
    meta["confirmed_by_3_high"] = confirmed_end

    qualifies = tail_len >= TAIL_MIN_INTERVALS
    if qualifies:
        prefix_range = float(np.max(np.ptp(q[:, :tail_len], axis=1)))
        qualifies = prefix_range < TAIL_Q_RANGE_MAX
        meta["prefix_q_range"] = prefix_range
    if qualifies:
        total_arc = float(s_a[-1])
        tail_arc = float(s_a[tail_len])
        fraction = tail_arc / total_arc if total_arc > 0 else 1.0
        meta["tail_arc_fraction"] = fraction
        if fraction > TAIL_MAX_FRACTION:
            # never auto-extend: whole branch to review, weights unchanged
            meta["tail_review"] = True
            meta["tail_rows"] = [int(v) for v in range(tail_len)]
            meta["reason"] = "tail_arc_exceeds_20pct"
        else:
            tail_flag[:tail_len] = True
            q_frame[:tail_len] = np.float32(TAIL_Q_WEIGHT)
            meta["tail_rows"] = [int(v) for v in range(tail_len)]
            meta["reason"] = "flat_tail_candidate"
    else:
        meta["reason"] = "no_qualifying_prefix"

    # contiguous low f OUTSIDE the tail -> uninformative non-tail slice
    run = 0
    for j in range(tail_len, n_frames):
        if low[j]:
            run += 1
            if run >= TAIL_MIN_INTERVALS:
                meta["uninformative_non_tail"] = True
                break
        else:
            run = 0
    return q_frame, tail_flag, meta


# ---------------------------------------------------------------------------
# first-contact teacher solver (ADR-0026 §2.3; plan §1.3)
# ---------------------------------------------------------------------------


def _quadratic_roots(a: float, b: float, c: float) -> list[float]:
    """Real roots of a x^2 + b x + c = 0 (float64, numerically careful)."""
    if abs(a) < 1e-300:
        if abs(b) < 1e-300:
            return []
        return [-c / b]
    disc = b * b - 4.0 * a * c
    if disc < 0.0:
        return []
    sqrt_disc = float(np.sqrt(disc))
    # numerically stable roots
    q = -0.5 * (b + np.sign(b if b != 0 else 1.0) * sqrt_disc)
    roots = []
    if q != 0.0:
        roots.append(float(q / a))
        roots.append(float(c / q))
    else:
        roots.append(float(-b / (2.0 * a)))
    return roots


def _earliest_root_in_segment(a: float, b: float, c: float) -> float | None:
    """Earliest root of the segment quadratic inside [0, 1] (tol ROOT_TOL)."""
    candidates = [
        r for r in _quadratic_roots(a, b, c)
        if -ROOT_TOL <= r <= 1.0 + ROOT_TOL
    ]
    if not candidates:
        return None
    return float(min(max(min(candidates), 0.0), 1.0))


FRAME_LAYER_REFERENCE = "reference_kabsch"
FRAME_LAYER_NATIVE_VERIFIED = "native_verified_v1"
#: Pre-registered continuity thresholds for native_verified_v1 stratification
#: (plan §1.3: rank-deficient references may fall back ONLY when the original
#: calculator archive demonstrably preserves one Cartesian frame).  Measured
#: consecutive-frame rotations/translations on failing branches are ~0.2-0.8°
#: / <=0.06 A — optimization-noise drift; a per-frame re-oriented archive
#: would show random tens-of-degrees rotations and fail these gates.
NATIVE_VERIFIED_MAX_CONSEC_ROT_DEG = 2.0
NATIVE_VERIFIED_MAX_CONSEC_TRANS_ANGSTROM = 0.10
NATIVE_VERIFIED_MAX_CUM_ROT_DEG = 10.0
NATIVE_VERIFIED_MAX_CUM_TRANS_ANGSTROM = 1.0


def verify_frame_continuity(pos: np.ndarray) -> dict:
    """Empirical native-frame continuity check (plan §1.3 escape hatch).

    Fits the optimal rigid transform between every CONSECUTIVE frame pair
    (unit weights, all atoms) and the cumulative transform frame T-1 vs
    frame 0.  The branch passes only when every consecutive rotation/
    translation and the cumulative one stay inside the pre-registered
    thresholds above — evidence that the archive never re-oriented the
    molecule between frames.  Evidence dict is sidecar-recorded.
    """
    x = np.asarray(pos, dtype=np.float64)
    n = len(x)
    if n < 2:
        return {"passed": False, "n_frames": int(n),
                "reason": "too_few_frames"}
    w = np.ones(x.shape[1], dtype=np.float64) / x.shape[1]
    angs: list[float] = []
    trs: list[float] = []

    def _rigid(A, B):
        _al, (R, muA, muB) = weighted_kabsch(A, B, w)
        ang = float(np.degrees(np.arccos(np.clip(
            (np.trace(R) - 1.0) / 2.0, -1.0, 1.0))))
        return ang, float(np.linalg.norm(muA - muB))

    for t in range(n - 1):
        ang, tr = _rigid(x[t + 1], x[t])
        angs.append(ang)
        trs.append(tr)
    cum_ang, cum_tr = _rigid(x[-1], x[0])
    checks = {
        "max_consecutive_rotation_deg": max(angs),
        "max_consecutive_translation_angstrom": max(trs),
        "cumulative_rotation_deg": cum_ang,
        "cumulative_translation_angstrom": cum_tr,
    }
    passed = (
        checks["max_consecutive_rotation_deg"]
        <= NATIVE_VERIFIED_MAX_CONSEC_ROT_DEG
        and checks["max_consecutive_translation_angstrom"]
        <= NATIVE_VERIFIED_MAX_CONSEC_TRANS_ANGSTROM
        and checks["cumulative_rotation_deg"]
        <= NATIVE_VERIFIED_MAX_CUM_ROT_DEG
        and checks["cumulative_translation_angstrom"]
        <= NATIVE_VERIFIED_MAX_CUM_TRANS_ANGSTROM)
    return {
        "passed": bool(passed),
        "n_frames": int(n),
        "thresholds": {
            "max_consecutive_rotation_deg":
                NATIVE_VERIFIED_MAX_CONSEC_ROT_DEG,
            "max_consecutive_translation_angstrom":
                NATIVE_VERIFIED_MAX_CONSEC_TRANS_ANGSTROM,
            "cumulative_rotation_deg": NATIVE_VERIFIED_MAX_CUM_ROT_DEG,
            "cumulative_translation_angstrom":
                NATIVE_VERIFIED_MAX_CUM_TRANS_ANGSTROM,
        },
        "evidence": {k: float(v) for k, v in checks.items()},
    }


def solve_teacher(
    pos: np.ndarray,
    key_rows: Sequence[int],
    ref_rows: Sequence[int],
    ref_weights: np.ndarray,
    d0: float,
    *,
    gauge_ok: bool = True,
    frame_layer: str = FRAME_LAYER_REFERENCE,
    cap: float = TEACHER_CAP_ANGSTROM,
) -> dict:
    """First-contact teacher labels for every start row t = 0..T-2.

    Future frames are aligned label-side into the X_t frame with the FIXED
    branch S/W (a weighted Kabsch per (t, j) using only reference rows);
    segments are piecewise-linear and include the first ``[X_t, Y_{t+1}]``.
    For each segment the contact equation ``RMS_A^2(Y(a) - X_t) = d0^2``
    (mean-square form of the ADR §2.3 quadratic; identical to ``D^2 = d0^2``
    for one key atom — the RMS reading is the one under which the per-atom
    cap can ever preempt contact) and the per-atom cap ``||y_i(a)-x_i|| =
    cap`` are quadratics in the segment parameter; the earliest root along
    the path wins (root tol 1e-8).  TS-terminal, capped, and
    no-future-step semantics follow ADR §2.3.  Rows are exactly ``0..T-2``;
    a row whose entire future is identical to X_t (within 1e-9) is exported
    ineligible with ``no_future_step`` — never a fabricated STOP label.

    Returns float64 arrays plus event diagnostics (``event_segment`` /
    ``event_alpha``); float32 casting happens only at shard write.
    """
    x_all = np.asarray(pos, dtype=np.float64)
    n_frames = len(x_all)
    key = np.asarray(key_rows, dtype=np.int64)
    ref = np.asarray(ref_rows, dtype=np.int64)
    weights = np.asarray(ref_weights, dtype=np.float64)
    if n_frames < 2:
        raise KeyAtomTeacherError("BRANCH_TOO_SHORT", f"T={n_frames}")
    if key.size == 0:
        raise KeyAtomTeacherError("NO_KEY_ATOMS", "empty key set")
    m = int(key.size)
    n_rows = n_frames - 1
    d0 = float(d0)

    out = {
        "start_row": np.arange(n_rows, dtype=np.int64),
        "u_raw": np.zeros((n_rows, m, 3), dtype=np.float64),
        "eligible": np.zeros(n_rows, dtype=bool),
        "reason_id": np.full(n_rows, REASON_OK, dtype=np.uint8),
        "terminal": np.zeros(n_rows, dtype=bool),
        "capped": np.zeros(n_rows, dtype=bool),
        "d_eff": np.zeros(n_rows, dtype=np.float64),
        "eta": np.zeros(n_rows, dtype=np.float64),
        "near_ts": np.zeros(n_rows, dtype=np.float64),
        "event_segment": np.full(n_rows, -1, dtype=np.int64),
        "event_alpha": np.full(n_rows, -1.0, dtype=np.float64),
    }

    ds = key_path_arc(x_all, key)
    s_a = np.concatenate([[0.0], np.cumsum(ds)])
    total_arc = float(s_a[-1])

    native_layer = (frame_layer == FRAME_LAYER_NATIVE_VERIFIED)
    if not gauge_ok and not native_layer:
        out["reason_id"].fill(REASON_GAUGE_UNRESOLVED)
        return out
    if not native_layer and ref.size == 0:
        out["reason_id"].fill(REASON_REFERENCE_INVALID)
        return out

    for t in range(n_rows):
        x_t = x_all[t]
        future = x_all[t + 1:]
        if float(np.max(np.abs(future - x_t))) <= FUTURE_STEP_TOL:
            out["reason_id"][t] = REASON_NO_FUTURE_STEP
            continue
        # label-side alignment of every future frame into the X_t frame with
        # the FIXED branch S/W (alignment uses only Y and X_t reference rows);
        # native_verified_v1 branches keep the archive's verified unified
        # Cartesian frame and skip the Kabsch entirely (plan §1.3).
        if native_layer:
            aligned = [x_t] + [x_all[j] for j in range(t + 1, n_frames)]
        else:
            aligned = [x_t]
            w_ref = weights[ref]
            for j in range(t + 1, n_frames):
                _a, transform = weighted_kabsch(x_all[j][ref], x_t[ref], w_ref)
                aligned.append(apply_rigid_transform(x_all[j], transform))
        x_key = x_t[key]

        event = None  # (kind, segment, alpha)
        for k in range(t, n_frames - 1):
            y0 = aligned[k - t][key]
            y1 = aligned[k - t + 1][key]
            delta = y1 - y0
            r0 = y0 - x_key
            # contact: RMS_A^2(a) = (1/m) sum_i ||r_i + a d_i||^2 = d0^2
            # (mean-square form of the ADR §2.3 quadratic; with m=1 it is
            # exactly D^2 = d0^2 — the RMS reading is the one under which
            # a per-atom cap can ever preempt contact, plan §1.3)
            a_q = float(np.sum(delta * delta)) / m
            b_q = 2.0 * float(np.sum(delta * r0)) / m
            c_q = float(np.sum(r0 * r0)) / m - d0 * d0
            contact = _earliest_root_in_segment(a_q, b_q, c_q)
            # per-atom cap: ||r_i + a d_i|| = cap, earliest atom
            cap_a = None
            for i in range(m):
                ai = float(delta[i] @ delta[i])
                bi = 2.0 * float(delta[i] @ r0[i])
                ci = float(r0[i] @ r0[i]) - cap * cap
                root = _earliest_root_in_segment(ai, bi, ci)
                if root is not None and (cap_a is None or root < cap_a):
                    cap_a = root
            seg_event = None
            if contact is not None and cap_a is not None:
                seg_event = ("cap" if cap_a < contact else "contact",
                             k, min(contact, cap_a))
            elif contact is not None:
                seg_event = ("contact", k, contact)
            elif cap_a is not None:
                seg_event = ("cap", k, cap_a)
            if seg_event is not None:
                event = seg_event
                break

        if event is None:
            # TS-terminal: path exhausted below d0
            y_ts = aligned[-1][key]
            u = y_ts - x_key
            out["u_raw"][t] = u
            out["terminal"][t] = True
            out["d_eff"][t] = float(np.sqrt(np.mean(np.sum(u * u, axis=1))))
            out["event_segment"][t] = n_frames - 2
            out["event_alpha"][t] = 1.0
        else:
            kind, k, alpha = event
            y0 = aligned[k - t][key]
            y1 = aligned[k - t + 1][key]
            y_alpha = y0 + alpha * (y1 - y0)
            u = y_alpha - x_key
            out["u_raw"][t] = u
            out["capped"][t] = kind == "cap"
            out["d_eff"][t] = float(np.sqrt(np.mean(np.sum(u * u, axis=1))))
            out["event_segment"][t] = k
            out["event_alpha"][t] = alpha
        out["eligible"][t] = True
        arc_left = total_arc - float(s_a[t])
        out["near_ts"][t] = (arc_left / total_arc) if total_arc > 0 else 0.0
        denom = arc_left
        out["eta"][t] = (
            float(np.linalg.norm(out["u_raw"][t])) / denom
            if denom > 1e-12 else 0.0)
    return out


# ---------------------------------------------------------------------------
# native + teacher shard writers
# ---------------------------------------------------------------------------


def d0_tag(d0: float) -> str:
    return f"{float(d0):g}"


def write_native_shard(
    out_root: str | Path, reaction_id: str, branch: int,
    pos: np.ndarray, key_rows: Sequence[int],
    q_frame: np.ndarray, key_progress: np.ndarray, tail_flag: np.ndarray,
    meta: Mapping,
) -> dict[str, str]:
    """Write ``native/<RXN>.branch{k}.npz`` + ``.meta.json`` sidecar.

    npz arrays (float64 except q_frame float32): pos [T,N,3] endpoint->TS,
    key_rows int64, q_frame [T], key_progress [T] (cumulative isolated
    key-atom arc fraction), tail_flag bool [T].  Returns artifact paths with
    sha256 digests.
    """
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    npz_path = root / f"{reaction_id}.branch{int(branch)}.npz"
    meta_path = root / f"{reaction_id}.branch{int(branch)}.meta.json"
    np.savez_compressed(
        npz_path,
        pos=np.asarray(pos, dtype=np.float64),
        key_rows=np.asarray(key_rows, dtype=np.int64),
        q_frame=np.asarray(q_frame, dtype=np.float32),
        key_progress=np.asarray(key_progress, dtype=np.float64),
        tail_flag=np.asarray(tail_flag, dtype=bool),
    )
    write_json(meta_path, dict(meta))
    return {
        "native_npz": {"path": str(npz_path), "sha256": sha256_file(npz_path)},
        "native_meta": {"path": str(meta_path), "sha256": sha256_file(meta_path)},
    }


def write_teacher_shard(
    teacher_root: str | Path, d0: float, reaction_id: str, branch: int,
    result: Mapping[str, np.ndarray],
) -> dict[str, str]:
    """Write ``teacher/d0_<v>/<RXN>.branch{k}.npz`` with EXACTLY the nine
    contract arrays (ADR-0026 §2.3): start_row int64 [S], u_raw float32
    [S,m,3], eligible bool [S], reason_id uint8 [S], terminal bool [S],
    capped bool [S], d_eff float32 [S], eta float32 [S], near_ts float32 [S].
    """
    root = Path(teacher_root) / f"d0_{d0_tag(d0)}"
    root.mkdir(parents=True, exist_ok=True)
    npz_path = root / f"{reaction_id}.branch{int(branch)}.npz"
    np.savez_compressed(
        npz_path,
        start_row=np.asarray(result["start_row"], dtype=np.int64),
        u_raw=np.asarray(result["u_raw"], dtype=np.float32),
        eligible=np.asarray(result["eligible"], dtype=bool),
        reason_id=np.asarray(result["reason_id"], dtype=np.uint8),
        terminal=np.asarray(result["terminal"], dtype=bool),
        capped=np.asarray(result["capped"], dtype=bool),
        d_eff=np.asarray(result["d_eff"], dtype=np.float32),
        eta=np.asarray(result["eta"], dtype=np.float32),
        near_ts=np.asarray(result["near_ts"], dtype=np.float32),
    )
    return {"path": str(npz_path), "sha256": sha256_file(npz_path)}


def build_branch_native(
    pos: np.ndarray, task: KeyAtomTask, branch: int,
) -> dict:
    """Assemble the native-branch record: reference, gate, tail, progress.

    Pure function of values (no I/O); the caller writes the shards.
    """
    x = np.asarray(pos, dtype=np.float64)
    ref = compute_reference_block(
        task.atomic_numbers, task.start_bonds(branch), task.key_atom_rows)
    # tiered gate: first candidate pool passing the scatter gate wins
    chosen = None
    tier_gates: list[dict] = []
    for tier in ref.get("tiers", [ref]):
        ok, eigenvalues = reference_gate(x[0], tier["rows"], tier["weights"])
        tier_gates.append({"tier": tier["tier"], "gate_ok": bool(ok),
                           "eigenvalues_angstrom2": eigenvalues,
                           "n_rows": int(np.asarray(tier["rows"]).size)})
        if ok and chosen is None:
            chosen = tier
    if chosen is None:
        chosen = ref.get("tiers", [ref])[-1]
        gate_ok, eigenvalues = False, tier_gates[-1][
            "eigenvalues_angstrom2"]
    else:
        gate_ok, eigenvalues = True, next(
            t["eigenvalues_angstrom2"] for t in tier_gates
            if t["tier"] == chosen["tier"])
    ref_rows = np.asarray(chosen["rows"], dtype=np.int64)
    ref_weights = np.asarray(chosen["weights"], dtype=np.float64)

    # native_verified_v1 stratification (plan §1.3): rank-deficient
    # reference blocks may keep the archive's verified unified Cartesian
    # frame instead — pre-registered continuity gates, evidence recorded.
    frame_layer = FRAME_LAYER_REFERENCE
    continuity: dict = {"passed": False, "reason": "not_attempted"}
    if not gate_ok:
        continuity = verify_frame_continuity(x)
        if continuity.get("passed"):
            frame_layer = FRAME_LAYER_NATIVE_VERIFIED
    k_pairs = task.K_pairs(branch)
    q_frame, tail_flag, tail_meta = mark_flat_tail(x, task.key_atom_rows, k_pairs)
    ds = key_path_arc(x, task.key_atom_rows)
    total = float(np.sum(ds))
    key_progress = np.concatenate([[0.0], np.cumsum(ds)])
    if total > 0:
        key_progress = key_progress / key_progress[-1]
    return {
        "pos": x,
        "key_rows": np.asarray(task.key_atom_rows, dtype=np.int64),
        "ref_rows": ref_rows,
        "ref_weights": ref_weights,
        "ref_hops": dict(chosen["hops"]),
        "ref_tier": str(chosen["tier"]),
        "ref_tier_gates": tier_gates,
        "n_rotor_bonds_cut": ref["n_rotor_bonds_cut"],
        "gate_ok": gate_ok,
        "frame_layer": frame_layer,
        "frame_continuity": continuity,
        "eigenvalues": eigenvalues,
        "k_pairs": k_pairs,
        "q_frame": q_frame,
        "tail_flag": tail_flag,
        "tail_meta": tail_meta,
        "key_progress": key_progress,
        "ds": ds,
    }


def native_meta_payload(
    task: KeyAtomTask, branch: int, record: Mapping, n_frames: int,
) -> dict:
    """Sidecar meta for one native branch shard (schema + gauge + tail +
    endpoint mapping evidence; never E/forces)."""
    tail_meta = record["tail_meta"]
    return {
        "schema_version": NATIVE_SCHEMA,
        "reaction_id": task.reaction_id,
        "branch": int(branch),
        "start_endpoint_index": int(branch),
        "start_endpoint_side": task.endpoint_sides[branch],
        "other_endpoint_side": task.endpoint_sides[1 - int(branch)],
        "endpoint_match_evidence": dict(task.endpoint_match_evidence),
        "n_frames": int(n_frames),
        "n_atoms": int(task.n_atoms),
        "key_rows": [int(r) for r in record["key_rows"]],
        "reference": {
            "rows": [int(r) for r in record["ref_rows"]],
            "weights": {str(int(r)): float(record["ref_weights"][r])
                        for r in record["ref_rows"]},
            "hops": {str(int(k)): int(v)
                     for k, v in record["ref_hops"].items()},
            "tier": str(record.get("ref_tier", "rigid_1_2hop")),
            "tier_gates": list(record.get("ref_tier_gates", [])),
            "n_rotor_bonds_cut": int(record["n_rotor_bonds_cut"]),
            "scatter_eigenvalues_angstrom2": [
                float(v) for v in record["eigenvalues"]],
            "gate_ok": bool(record["gate_ok"]),
            "gate_rule": ("lambda1 > 0.01 A^2 and lambda2/lambda1 > 1e-3; "
                          "planar OK; else gauge_unresolved; tiered fallback "
                          "rigid_1_2hop -> any_1_2hop -> any_3hop -> "
                          "all_nonkey_heavy per plan §1.3 block-size rule"),
        },
        "frame_layer": str(record.get("frame_layer", "reference_kabsch")),
        "frame_continuity": dict(record.get("frame_continuity", {})),
        "tail": {
            "flagged_rows": [int(v) for v in
                             np.nonzero(record["tail_flag"])[0]],
            "q_weight_in_tail": TAIL_Q_WEIGHT,
            "max_arc_fraction": TAIL_MAX_FRACTION,
            "review": bool(tail_meta.get("tail_review", False)),
            "reason": str(tail_meta.get("reason", "")),
            "prefix_q_range": tail_meta.get("prefix_q_range"),
            "tail_arc_fraction": tail_meta.get("tail_arc_fraction"),
            "uninformative_whole": bool(tail_meta.get(
                "uninformative_whole", False)),
            "uninformative_non_tail": bool(tail_meta.get(
                "uninformative_non_tail", False)),
        },
        "sources": {k: dict(v) for k, v in task.sources.items()},
        "notes": ("frames are the native endpoint->TS branch as stored; "
                  "isolated per branch, no cross-TS continuation; no "
                  "energy/gradient values are stored"),
    }


# ---------------------------------------------------------------------------
# conflict audit (plan §1.2 bounded: fit50-internal)
# ---------------------------------------------------------------------------


def _atom_invariants(record: Mapping) -> dict[int, tuple]:
    """Per-atom invariant class key: (element, start-degree, is_key, ring)."""
    n = len(record["atomic_numbers"])
    key = set(int(r) for r in record["key_rows"])
    out: dict[int, tuple] = {}
    for i in range(n):
        out[i] = (
            int(record["atomic_numbers"][i]),
            int(record["deg_start"][i]),
            i in key,
            bool(record["is_ring"][i]),
        )
    return out


def _orbit_permutations(record_a: Mapping, record_b: Mapping):
    """Enumerate allowed permutations of B's atoms onto A's atoms within
    per-atom invariant classes (element/degree/is_key/ring).  Returns
    ``(perms, overflow)`` where each perm ``p`` means atom i of A pairs with
    atom p[i] of B.  Enumeration is capped at ORBIT_ENUM_CAP combinations;
    overflow returns no permutations (plan §1.2: never a random subset)."""
    inv_a = _atom_invariants(record_a)
    inv_b = _atom_invariants(record_b)
    classes: dict[tuple, list[int]] = {}
    for i in sorted(inv_a):
        classes.setdefault(inv_a[i], []).append(i)
    b_classes: dict[tuple, list[int]] = {}
    for i in sorted(inv_b):
        b_classes.setdefault(inv_b[i], []).append(i)
    if set(classes) != set(b_classes):
        return [], False  # invariant structure differs: not orbit-comparable
    from itertools import permutations, product

    total = 1
    for key, members in classes.items():
        if len(b_classes[key]) != len(members):
            return [], False
        total *= _factorial(len(members))
        if total > ORBIT_ENUM_CAP:
            return [], True
    slots = []
    for key in sorted(classes):
        slots.append(list(permutations(b_classes[key])))
    perms = []
    for combo in product(*slots):
        perm = [0] * len(inv_a)
        for class_members, assigned in zip(
                (classes[k] for k in sorted(classes)), combo):
            for i, b_idx in zip(class_members, assigned):
                perm[i] = b_idx
        perms.append(tuple(perm))
        if len(perms) > ORBIT_ENUM_CAP:
            return [], True
    return perms, False


def _factorial(n: int) -> int:
    out = 1
    for v in range(2, n + 1):
        out *= v
    return out


def _flatten_cosine(u_a: np.ndarray, u_b: np.ndarray) -> float:
    a = np.asarray(u_a, dtype=np.float64).ravel()
    b = np.asarray(u_b, dtype=np.float64).ravel()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(a @ b / (na * nb))


def conflict_audit(branch_records: Sequence[Mapping]) -> tuple[dict, dict]:
    """Bounded fit50-internal conflict audit (plan §1.2; ADR §2.5).

    ``branch_records`` carry per-branch X0 geometry, invariants, and the
    main-d0 teacher label at start row 0 (u0).  Candidate condition: min over
    the automorphism orbit of full-molecule RMSD < 0.03 A AND key RMSD <
    0.02 A (X0 proper-rotation alignment).  Final conflict additionally
    requires the same branch index, label RMS > 0.02 A, and flattened
    cosine C < 0; C in [0, 0.25) is a near-conflict (weight 0.25).
    Conflict connected components without a pre-registered bad-map rule get
    weight 0 as a whole.  Orbit enumeration above 128 combinations is
    ``mapping_group_overflow`` — the pair is skipped and recorded, never
    approximated with a random subset.

    Returns ``(audit_doc, weights)`` with
    ``weights = {reaction_id: {branch: weight}}``.
    """
    records = list(branch_records)
    outcomes = {
        "size_mismatch": 0,
        "invariant_mismatch": 0,
        "mapping_group_overflow": 0,
        "far": 0,
        "context_resolved": 0,
        "near_conflict": 0,
        "conflict": 0,
    }
    overflow_pairs: list[dict] = []
    uf = _UnionFindAudit()
    conflict_edges: list[tuple[int, int]] = []
    near_edges: list[tuple[int, int]] = []
    context_pairs = 0

    n = len(records)
    for a in range(n):
        for b in range(a + 1, n):
            ra, rb = records[a], records[b]
            if len(ra["atomic_numbers"]) != len(rb["atomic_numbers"]):
                outcomes["size_mismatch"] += 1
                continue
            perms, overflow = _orbit_permutations(ra, rb)
            if overflow:
                outcomes["mapping_group_overflow"] += 1
                overflow_pairs.append({
                    "a": f"{ra['reaction_id']}.branch{ra['branch']}",
                    "b": f"{rb['reaction_id']}.branch{rb['branch']}",
                    "reason": "mapping_group_overflow",
                })
                continue
            if not perms:
                outcomes["invariant_mismatch"] += 1
                continue
            key_a = np.asarray(ra["key_rows"], dtype=np.int64)
            key_b_set = set(int(v) for v in rb["key_rows"])
            pos_a = np.asarray(ra["pos0"], dtype=np.float64)
            pos_b = np.asarray(rb["pos0"], dtype=np.float64)
            candidate = None
            for perm in perms:
                # key correspondence required for the key-RMSD criterion
                if {perm[int(i)] for i in key_a} != key_b_set:
                    continue
                permuted = pos_b[[perm[i] for i in range(len(pos_a))]]
                aligned, _ = weighted_kabsch(
                    permuted, pos_a, np.ones(len(pos_a)))
                rmsd_full = float(np.sqrt(np.mean(np.sum(
                    (aligned - pos_a) ** 2, axis=1))))
                if rmsd_full >= CONFLICT_FULL_RMSD:
                    continue
                rmsd_key = float(np.sqrt(np.mean(np.sum(
                    (aligned[key_a] - pos_a[key_a]) ** 2, axis=1))))
                if rmsd_key >= CONFLICT_KEY_RMSD:
                    continue
                candidate = (perm, rmsd_full)
                break
            if candidate is None:
                outcomes["far"] += 1
                continue
            perm, _rmsd = candidate
            if int(ra["branch"]) != int(rb["branch"]):
                outcomes["context_resolved"] += 1
                context_pairs += 1
                continue
            u_a = np.asarray(ra["u0"], dtype=np.float64)
            u_b_full = np.asarray(rb["u0"], dtype=np.float64)
            if u_a.shape != u_b_full.shape:
                outcomes["context_resolved"] += 1
                context_pairs += 1
                continue
            # reorder B's label rows into A's key-row order via the perm
            key_order_b = [perm[int(i)] for i in key_a]
            pos_of = {int(r): k for k, r in enumerate(rb["key_rows"])}
            u_b = np.stack([u_b_full[pos_of[r]] for r in key_order_b])
            label_rms = float(np.sqrt(np.mean((u_a - u_b) ** 2)))
            cos = _flatten_cosine(u_a, u_b)
            if label_rms > CONFLICT_LABEL_RMS and cos < CONFLICT_COSINE_NEGATIVE:
                outcomes["conflict"] += 1
                conflict_edges.append((a, b))
                uf.union(a, b)
            elif (label_rms > CONFLICT_LABEL_RMS
                  and CONFLICT_COSINE_NEGATIVE <= cos < NEAR_CONFLICT_COSINE):
                outcomes["near_conflict"] += 1
                near_edges.append((a, b))
            else:
                outcomes["context_resolved"] += 1
                context_pairs += 1

    # component weights: whole conflict component -> 0; near -> 0.25
    weights_list = [1.0] * n
    zero_component: dict[str, list[int]] = {}
    for a, b in conflict_edges:
        root = uf.find(a)
        zero_component.setdefault(root, []).extend([a, b])
    zero_idx: set[int] = set()
    for members in zero_component.values():
        zero_idx.update(members)
    for i in zero_idx:
        weights_list[i] = ZERO_WEIGHT
    for a, b in near_edges:
        if weights_list[a] > 0:
            weights_list[a] = NEAR_CONFLICT_WEIGHT
        if weights_list[b] > 0:
            weights_list[b] = NEAR_CONFLICT_WEIGHT

    weights: dict[str, dict[str, float]] = {}
    for i, record in enumerate(records):
        weights.setdefault(str(record["reaction_id"]), {})[
            str(record["branch"])] = float(weights_list[i])

    usable = [i for i in range(n) if weights_list[i] > 0]
    conflicted_usable = {
        uf.find(a) for a, _b in conflict_edges
    }
    residual = 0
    for i in usable:
        if uf.find(i) in conflicted_usable and i not in zero_idx:
            residual += 1
    residual_rate = residual / len(usable) if usable else 0.0
    isolation_rate = len(zero_idx) / n if n else 0.0

    audit = {
        "schema_version": CONFLICT_AUDIT_SCHEMA,
        "scope": "fit50_internal",
        "note": ("bounded audit over the 50 fit groups (100 branches); the "
                 "historical 342 pairs were NOT re-scanned in this window "
                 "(declared evidence boundary, ADR-0026 §6)"),
        "thresholds": {
            "candidate_full_rmsd_angstrom": CONFLICT_FULL_RMSD,
            "candidate_key_rmsd_angstrom": CONFLICT_KEY_RMSD,
            "label_rms_angstrom": CONFLICT_LABEL_RMS,
            "conflict_cosine": f"C < {CONFLICT_COSINE_NEGATIVE}",
            "near_conflict_cosine": (
                f"{CONFLICT_COSINE_NEGATIVE} <= C < {NEAR_CONFLICT_COSINE}"),
            "orbit_enum_cap": ORBIT_ENUM_CAP,
            "orbit_invariants": ["element", "degree_start", "is_key", "ring"],
        },
        "n_branches": n,
        "n_pairs": n * (n - 1) // 2,
        "pair_outcomes": outcomes,
        "overflow_pairs": overflow_pairs,
        "residual_conflict_rate": residual_rate,
        "isolation_rate": isolation_rate,
        "zero_weight_branches": sorted(
            f"{records[i]['reaction_id']}.branch{records[i]['branch']}"
            for i in zero_idx),
        "component_weights": weights,
    }
    return audit, weights


class _UnionFindAudit:
    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    def find(self, x: int) -> int:
        p = self._parent.setdefault(x, x)
        if p != x:
            p = self.find(p)
            self._parent[x] = p
        return p

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            hi, lo = max(ra, rb), min(ra, rb)
            self._parent[hi] = lo


def write_conflict_artifacts(
    audit: Mapping, weights: Mapping,
    audit_path: str | Path, weights_path: str | Path,
) -> dict[str, str]:
    """Write ``dataset_audit.json`` + ``conflict_weights.json``."""
    audit_path = Path(audit_path)
    weights_path = Path(weights_path)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(audit_path, dict(audit))
    payload = {
        "schema_version": CONFLICT_AUDIT_SCHEMA,
        "weights": {str(rid): {str(b): float(w) for b, w in branches.items()}
                    for rid, branches in dict(weights).items()},
        "note": ("training-side conflict weights; fit50-internal audit "
                 "scope (plan §1.2)"),
    }
    write_json(weights_path, payload)
    return {
        "dataset_audit": {"path": str(audit_path),
                          "sha256": sha256_file(audit_path)},
        "conflict_weights": {"path": str(weights_path),
                             "sha256": sha256_file(weights_path)},
    }


# ---------------------------------------------------------------------------
# frame contract (ADR-0026 §2.2)
# ---------------------------------------------------------------------------


def build_frame_contract() -> dict:
    """The V5 frame contract document (schema pesn_v5_frame_contract_v1)."""
    return {
        "schema_version": FRAME_CONTRACT_SCHEMA,
        "adr": "ADR-0026 §2.2",
        "plan_sections": ["§1.1", "§1.3"],
        "input_frame": ("causal current frame X_t: the native branch frame "
                        "as stored, no future alignment of any kind"),
        "anchor_frame": ("X0 = branch start row 0 aligned into the X_t frame "
                         "by the fixed S/W weighted Kabsch using only X0 and "
                         "X_t (feature-lane concern; recorded here)"),
        "label_frame": ("X_t frame; future frames Y[j] are aligned into X_t "
                        "with the SAME fixed branch S/W on the label side "
                        "only, then interpolated piecewise-linearly"),
        "reference_policy": {
            "set": ("start-endpoint non-key heavy atoms within 1-2 covalent "
                    "hops of the key set, in the rigid block(s) after cutting "
                    "non-ring single bonds with both endpoints degree>=2"),
            "weights": "w_i = [i in S] / (1 + hop(i,A))^2; W = diag(w/sum w)",
            "gate": ("lambda1 > 0.01 A^2 and lambda2/lambda1 > 1e-3 on the "
                     "start-frame scatter; planar references allowed (no "
                     "third-eigenvalue requirement); failure -> "
                     "gauge_unresolved, never a global axis fallback"),
            "tiered_fallback": ("rigid_1_2hop -> any_1_2hop -> any_3hop -> "
                                "all_nonkey_heavy (plan §1.3 candidate-block "
                                "distance/size rule); first tier passing the "
                                "scatter gate wins; per-branch tier and all "
                                "tier gates recorded in the native sidecar"),
            "native_verified_v1": (
                "branches whose every tier fails the scatter gate may keep "
                "the archive's unified Cartesian frame INSTEAD of Kabsch "
                "alignment, only after verify_frame_continuity passes the "
                "pre-registered thresholds (consecutive rotation <= 2.0 deg, "
                "translation <= 0.10 A; cumulative <= 10.0 deg / 1.0 A); "
                "evidence per branch in the native sidecar; failing "
                "branches stay gauge_unresolved (plan §1.3)"),
        },
        "kabsch": {
            "convention": "row vectors; C=(Y-muY)^T W (X-muX)=U S V^T",
            "rotation": "R = U diag(1,1,det(UV^T)) V^T (proper only)",
            "aligned": "(Y-muY) R + muX",
            "dtype": "float64",
        },
        "interpolation": ("piecewise-linear on aligned Cartesian coordinates "
                          "per segment; segments include the first "
                          "[X_t, Y_{t+1}]; no rotation interpolation, no "
                          "Cartesian cubic spline, no SLERP"),
        "root_finding": {
            "contact": ("per segment solve sum_i ||y_i(a)-x_i||^2 = d0^2; "
                        "earliest root in [0,1] along the path; float64"),
            "root_tolerance": ROOT_TOL,
        },
        "cap_angstrom": TEACHER_CAP_ANGSTROM,
        "d0_ablation_angstrom": list(D0_VALUES),
        "no_future_step": ("a start row whose entire future is identical to "
                           "X_t within 1e-9 gets no MOVE label "
                           "(reason no_future_step); STOP is never fabricated"),
    }


# ---------------------------------------------------------------------------
# stage-0 orchestration (guarded; resumable)
# ---------------------------------------------------------------------------


def load_frames_via_truth_bridge(
    reaction_id: str, config: Mapping, allow_truth: bool,
) -> np.ndarray:
    """Audited IRC coordinates for one fit reaction (coordinates ONLY).

    Frames arrive through ``truth_source.load_scoped_irc_frames`` — the
    registered bridge enforces ``allow_truth`` and frozen-split membership
    before any path is resolved, and its audit log records the bridge module
    as the caller (the guard allowlist admits only that module; the
    campaign's ``paths.truth_caller`` id is recorded in artifact metadata).
    EHG/forces stay inside the bridge payload and are never copied out.
    """
    from pes2ts_core.pesn.offline.data.truth_source import load_scoped_irc_frames

    paths = config.get("paths") or {}
    payload = load_scoped_irc_frames(
        reaction_id,
        allow_truth=allow_truth,
        manifests_dir=paths.get("manifests_dir", "data/manifests"),
        split_dir=paths.get(
            "split_dir", "outputs/pesn/direct_egnn_v1_20261015/splits"),
    )
    coordinates = payload.get("coordinates")
    if coordinates is None:
        raise KeyAtomTeacherError(
            "FRAMES_MISSING", f"no coordinates for {reaction_id}")
    return np.asarray(coordinates, dtype=np.float64)


def _p1_path_for(p1_root: str | Path, reaction_id: str) -> Path:
    number = int(reaction_id.split("_")[-1])
    return Path(p1_root) / f"{number // 1000:05d}" / f"{reaction_id}.json"


def _branch_frames_for_task(
    task: KeyAtomTask, ordered_path: np.ndarray, ts_position: int,
) -> dict[int, np.ndarray]:
    """Map IRC R/P branches onto task branch indices via endpoint sides."""
    branch_r, branch_p = split_dual_branches(ordered_path, ts_position)
    out: dict[int, np.ndarray] = {}
    for b in (0, 1):
        side = task.endpoint_sides[b]
        out[b] = branch_p if side == "P" else branch_r
    return out


def build_reaction_branches(
    task: KeyAtomTask, frames: np.ndarray, p1_doc: Mapping,
) -> dict[int, dict]:
    """Full per-reaction teacher build from plain frame values.

    Returns ``{branch: record}`` where each record bundles the native-branch
    record plus the three d0 teacher results (float64, pre-cast).
    """
    validation = p1_doc.get("irc_validation") or {}
    layout = str(validation.get("layout", ""))
    try:
        ordered, _energies, ts_position = order_frames_and_energy(
            frames, None,
            layout=layout,
            orientation=str(validation.get("orientation", "")),
            ts_frame_index=int(validation.get("ts_frame_index", -1)),
            splice_index=validation.get("splice_index"),
        )
    except DualTSPreprocessError as exc:
        raise KeyAtomTeacherError(exc.code, exc.detail) from exc
    branch_of = _branch_frames_for_task(task, ordered, ts_position)
    records: dict[int, dict] = {}
    for b in (0, 1):
        pos = branch_of[b]
        if pos.shape[1] != task.n_atoms:
            raise KeyAtomTeacherError(
                "ATOM_COUNT_MISMATCH",
                f"{task.reaction_id} branch {b}: frames have {pos.shape[1]} "
                f"atoms, task has {task.n_atoms}")
        record = build_branch_native(pos, task, b)
        record["teachers"] = {
            d0: solve_teacher(
                pos, task.key_atom_rows, record["ref_rows"],
                record["ref_weights"], d0,
                gauge_ok=record["gate_ok"],
                frame_layer=record.get(
                    "frame_layer", FRAME_LAYER_REFERENCE))
            for d0 in D0_VALUES
        }
        records[b] = record
    return records


def _reaction_is_complete(
    native_meta_path: Path, teacher_root: Path, task: KeyAtomTask,
) -> bool:
    """Resumability check: shards exist and recorded source hashes match."""
    if not native_meta_path.is_file():
        return False
    try:
        meta = read_json(native_meta_path)
    except Exception:
        return False
    for name, entry in task.sources.items():
        recorded = (meta.get("sources") or {}).get(name) or {}
        if recorded.get("sha256") != entry.get("sha256"):
            return False
    for d0 in D0_VALUES:
        shard = (Path(teacher_root) / f"d0_{d0_tag(d0)}"
                 / f"{task.reaction_id}.branch{meta.get('branch')}.npz")
        if not shard.is_file():
            return False
    return True


def _load_branch_record_from_shards(
    task: KeyAtomTask, branch: int, native_root: Path,
) -> dict:
    """Reload a previously written branch for the audit stage (resume)."""
    npz_path = Path(native_root) / f"{task.reaction_id}.branch{branch}.npz"
    with np.load(npz_path) as data:
        pos = np.asarray(data["pos"], dtype=np.float64)
        key_rows = np.asarray(data["key_rows"], dtype=np.int64)
    teacher_main = (Path(native_root).parent / "teacher"
                    / f"d0_{d0_tag(MAIN_D0)}"
                    / f"{task.reaction_id}.branch{branch}.npz")
    with np.load(teacher_main) as data:
        u0 = np.asarray(data["u_raw"][0], dtype=np.float64)
    return _audit_branch_record(task, branch, pos, key_rows, u0)


def _audit_branch_record(
    task: KeyAtomTask, branch: int, pos: np.ndarray, key_rows: np.ndarray,
    u0: np.ndarray,
) -> dict:
    """Assemble one conflict-audit branch record (X0 + invariants + label)."""
    n = task.n_atoms
    adj = _adjacency(n, task.start_bonds(branch))
    rings = _ring_bonds(n, task.start_bonds(branch))
    key_set = set(int(r) for r in key_rows)
    return {
        "reaction_id": task.reaction_id,
        "branch": int(branch),
        "pos0": np.asarray(pos[0], dtype=np.float64),
        "key_rows": np.asarray(key_rows, dtype=np.int64),
        "u0": np.asarray(u0, dtype=np.float64),
        "atomic_numbers": [int(z) for z in task.atomic_numbers],
        "deg_start": [len(adj[i]) for i in range(n)],
        "is_ring": [i in rings for i in range(n)],
        "is_key": [i in key_set for i in range(n)],
    }


def prepare_audit(config: Mapping, *, allow_truth: bool = False) -> dict:
    """Stage-0 orchestrator: tasks, native shards, teachers, audits.

    Guarded: refuses when the frozen V5 split manifest is missing (the split
    identity must exist before any label work — plan §1 build order).
    Sequential and resumable: reactions whose shards already exist with
    matching source hashes are skipped and re-loaded for the audit.
    Writes a progress json with the sha256 of every artifact.
    """
    paths = config.get("paths") or {}
    split_manifest = Path(paths.get(
        "split_manifest", "data/manifests/v5/split_manifest.json"))
    if not split_manifest.is_file():
        raise KeyAtomTeacherError(
            "SPLIT_MANIFEST_MISSING",
            f"frozen V5 split manifest not found at {split_manifest}; run "
            "pes2ts_core.pesn.offline.splits.v5_fold --write first")
    fit50_selection = Path(paths.get(
        "fit50_selection",
        "outputs/pesn/direct_egnn_v1_20261015/splits/selection_direct_fit50.json"))
    selection = read_json(fit50_selection)
    reaction_ids = [str(r) for r in selection.get("reaction_ids") or []]

    wp2_root = Path(paths.get(
        "wp2_task_root", "outputs/pesn/dual_ts_v2_20261015/wp2_fit50"))
    p1_root = Path(paths.get(
        "p1_mapping_root", "data/interim/g1_truth/p1_mapping"))
    native_root = Path(paths.get("native_root", "outputs/pesn/v5/native"))
    teacher_root = Path(paths.get("teacher_root", "outputs/pesn/v5/teacher"))
    manifests_dir = Path(paths.get("manifests_dir", "data/manifests"))
    task_root = Path(paths.get("task_root", "outputs/pesn/v5/tasks"))
    output_root = Path(paths.get("output_root", "outputs/pesn/v5/sprint"))

    progress: dict = {
        "schema_version": TEACHER_PROGRESS_SCHEMA,
        "campaign_id": (config.get("sprint") or {}).get(
            "campaign_id", "V5_SPRINT"),
        "truth_caller": str(paths.get("truth_caller", "")),
        "split_manifest": {
            "path": str(split_manifest), "sha256": sha256_file(split_manifest)},
        "reactions": {},
        "stage_artifacts": {},
    }
    audit_records: list[dict] = []
    for reaction_id in reaction_ids:
        wp2_path = wp2_root / f"{reaction_id}.task.json"
        p1_path = _p1_path_for(p1_root, reaction_id)
        task = build_key_atom_task(
            read_json(wp2_path), read_json(p1_path),
            wp2_path=str(wp2_path), p1_path=str(p1_path))
        task_path = task_root / f"{reaction_id}.task.json"
        entry: dict = {"status": "built", "artifacts": {}}
        branch_meta_paths = [
            native_root / f"{reaction_id}.branch{b}.meta.json" for b in (0, 1)]
        if all(_reaction_is_complete(p, teacher_root, task)
               for p in branch_meta_paths):
            entry["status"] = "skipped_complete"
        else:
            frames = load_frames_via_truth_bridge(
                reaction_id, config, allow_truth)
            p1_doc = read_json(p1_path)
            records = build_reaction_branches(task, frames, p1_doc)
            for b, record in records.items():
                meta = native_meta_payload(task, b, record, len(record["pos"]))
                entry["artifacts"].update(write_native_shard(
                    native_root, reaction_id, b, record["pos"],
                    task.key_atom_rows, record["q_frame"],
                    record["key_progress"], record["tail_flag"], meta))
                for d0 in D0_VALUES:
                    entry["artifacts"][f"teacher_d0_{d0_tag(d0)}_b{b}"] = \
                        write_teacher_shard(
                            teacher_root, d0, reaction_id, b,
                            record["teachers"][d0])
        if not task_path.is_file():
            task.save(task_path)
        entry["artifacts"]["task"] = {
            "path": str(task_path), "sha256": sha256_file(task_path)}
        # audit records: fresh or reloaded (resume path)
        for b in (0, 1):
            if entry["status"] == "skipped_complete":
                record = _load_branch_record_from_shards(
                    task, b, native_root)
            else:
                rec = records[b]  # type: ignore[possibly-undefined]
                record = _audit_branch_record(
                    task, b, rec["pos"], rec["key_rows"],
                    rec["teachers"][MAIN_D0]["u_raw"][0])
            audit_records.append(record)
        progress["reactions"][reaction_id] = entry

    audit, weights = conflict_audit(audit_records)
    progress["stage_artifacts"].update(write_conflict_artifacts(
        audit, weights,
        manifests_dir / "v5" / "dataset_audit.json",
        teacher_root.parent / "conflict_weights.json"))
    frame_contract = build_frame_contract()
    frame_path = manifests_dir / "v5" / "frame_contract.json"
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(frame_path, frame_contract)
    progress["stage_artifacts"]["frame_contract"] = {
        "path": str(frame_path), "sha256": sha256_file(frame_path)}
    progress_path = output_root / "prepare_audit_progress.json"
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(progress_path, progress)
    return progress


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="V5 stage-0: key-atom teacher prepare/audit")
    parser.add_argument("--config", default="config/campaigns/v5_sprint.yaml")
    parser.add_argument(
        "--stage", choices=["prepare-audit"], required=True)
    parser.add_argument(
        "--allow-truth", action="store_true",
        help="audited IRC access gate (same discipline as the v1 pipeline)")
    args = parser.parse_args(argv)

    import yaml  # local import: only the CLI needs YAML

    config_path = Path(args.config)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if args.stage == "prepare-audit":
        paths = config.get("paths") or {}
        split_manifest = Path(paths.get(
            "split_manifest", "data/manifests/v5/split_manifest.json"))
        if not split_manifest.is_file():
            print(f"refusing: frozen V5 split manifest missing at "
                  f"{split_manifest}")
            return 2
        progress = prepare_audit(config, allow_truth=args.allow_truth)
        built = sum(1 for e in progress["reactions"].values()
                    if e["status"] == "built")
        skipped = sum(1 for e in progress["reactions"].values()
                      if e["status"] == "skipped_complete")
        print(f"prepare-audit done: {built} built, {skipped} skipped "
              f"(resumable), {len(progress['reactions'])} reactions total")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
