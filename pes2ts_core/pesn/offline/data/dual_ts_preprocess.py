"""Dual-endpoint TS teacher preprocessing small package (WP1, plan v4 §D/§I.4).

Builds, for ONE reaction group, the two unordered endpoint graphs and the
complete dual-branch TS teacher record (A→TS / B→TS) from an IRC trajectory,
using the SAME audited truth-access path as the v1 direct preprocessing
(``truth_source`` bridge, ``--allow-truth`` gated) and the SAME audited
rigid-drift registration (``direct_preprocess.register_path``).

WP1 scope (G1 release conditions C5 + WP2 checklist honored):

* steps 1-4 of the plan §I.4 fixed pipeline that are geometry work are done
  here: identity/mapping check, two-sided endpoint→TS ordering, rigid drift
  removal, and label generation.  Step 5 (node insertion on a_max exceedance)
  and step 7 (physical observation completion) are WP2 scope — in WP1 the
  exceedances and missing observations are only FLAGGED, item by item, in the
  gaps registry; nothing is inserted, resampled, or fetched.
* the TS state of each branch is the terminal ``teacher_endpoint``: no
  successor, zero remaining displacement, no action — the two branches are
  never chained across the TS.
* labels are computed by differencing the EXPORTED (rounded) coordinates:
  ``a_t = X_{t+1} - X_t``, ``d_t = X* - X``, ``ell`` = remaining arc length
  under the unit-mass metric bound to
  :data:`~pes2ts_core.pesn.contracts.dual_ts.ARC_LENGTH_METRIC_VERSION`
  (real mass metric is deferred; C5 discipline).
* endpoint graphs come from the G1 P1 authority documents (v1/G1's bond
  source: ``r_bonds``/``p_bonds`` per endpoint, bond orders bound to their
  edge and endpoint).  When the authority document is unavailable entirely,
  a deterministic covalent-radius single-link perception is used instead and
  recorded in the task ``method_id``/provenance.
* unordered identity: the endpoint pair is canonically ordered by content
  digest (:func:`canonical_endpoint_order`), so an A/B swap yields an
  identical task content digest (verified at build time).
* the task↔teacher cross-record identity check (WP2 checklist item done now)
  verifies every teacher state's atomic numbers and atom mapping against the
  task endpoints; mismatch is a typed rejection.

No piece of this module inserts nodes, clips actions, fabricates
observations, or touches any quarantined file directly: IRC frames arrive as
plain values from the audited bridge, exactly as in the v1 pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.contracts.dual_ts import (
    ARC_LENGTH_METRIC_VERSION,
    BOUNDARY_TEACHER_ENDPOINT,
    DualEndpointTask,
    DualTSTeacher,
    EndpointBond,
    EndpointGraph,
    EpisodeRecord,
    BranchRecord,
    TeacherLabel,
    TeacherState,
    canonical_endpoint_order,
)
from pes2ts_core.pesn.offline.data.direct_preprocess import (
    masses_for,
    register_path,
)

DUAL_TS_PREPROCESS_VERSION: Final[str] = "pes2ts_dual_ts_preprocess_wp1_v0"

#: First-round per-atom action bound (plan v4 §I.8): exceedances are flagged
#: per step in the gaps registry — WP1 never inserts nodes (step 5 is WP2).
A_MAX_ANGSTROM: Final[float] = 0.1

#: C5 frozen metric (WP2): diagonal-mass arc-length metric
#: ``ell_t = sum_{k>=t} ||X_{k+1} - X_k||_M`` with
#: ``||delta||_M = sqrt(sum_i m_i ||delta_i||^2)``, masses taken from the
#: SAME v1 table :func:`direct_preprocess.masses_for` uses.  This is a NEW
#: binding constant — the WP1 placeholder
#: (``contracts.dual_ts.ARC_LENGTH_METRIC_VERSION`` = ``..._v0_PLACEHOLDER``)
#: stays untouched so WP1 artifacts keep their recorded version.
ARC_LENGTH_METRIC_V1_DIAG_MASSES: Final[str] = "mass_weighted_arc_v1_diag_masses"

#: Export precision of coordinates/labels; labels are differenced from the
#: exported values (C1 discipline), so the contract's 1e-6 identities hold.
EXPORT_COORD_DECIMALS: Final[int] = 10

#: Bond perception provenance strings (recorded in the task method_id).
BOND_PERCEPTION_AUTHORITY: Final[str] = "g1_p1_mapping_endpoint_bonds_v1"
BOND_PERCEPTION_COVALENT: Final[str] = "covalent_radius_single_link_v0"

#: TS-frame identification rules (recorded per reaction).
TS_RULE_AUTHORITY: Final[str] = "g1_irc_validation_ts_frame_v1"
TS_RULE_MAX_ENERGY: Final[str] = "max_energy_frame_v0"

#: Rigid drift rule: v1's chained mass-weighted Kabsch registration
#: (translation + rotation, det +1, no reflection) — reused, not reimplemented.
DRIFT_RULE: Final[str] = "chained_mass_weighted_kabsch_rigid_v1"

#: Single-link covalent-radius tolerance (Å) for the fallback perception.
COVALENT_BOND_TOL: Final[float] = 0.45

#: WP1 observation policy: xTB E/g is WP2 scope — every state's observation
#: is null + a registered reason, never a zero-filled placeholder.
DEFAULT_OBS_MISSING_REASON: Final[str] = "observation_not_exported"
DEFAULT_TS_OBS_MISSING_REASON: Final[str] = "teacher_frame_isolated"

#: Cordero 2008 single-bond covalent radii (Å) for the fallback perception.
COVALENT_RADII: Final[Mapping[int, float]] = {
    1: 0.31, 2: 0.28, 3: 1.28, 4: 0.96, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66,
    9: 0.57, 10: 0.58, 11: 1.66, 12: 1.41, 13: 1.21, 14: 1.11, 15: 1.07,
    16: 1.05, 17: 1.02, 18: 1.06, 19: 2.03, 20: 1.76, 26: 1.32, 27: 1.26,
    28: 1.24, 29: 1.32, 30: 1.22, 34: 1.20, 35: 1.20, 53: 1.39,
}

ARC_IDENTITY_TOL: Final[float] = 1e-8


class DualTSPreprocessError(RuntimeError):
    """Typed preprocessing refusal (never silently drops a reaction)."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# Audited IRC access (same bridge as v1; lazy import keeps pesn modules clean)
# ---------------------------------------------------------------------------


def load_irc_frames_for_reaction(
    reaction_id: str,
    *,
    allow_truth: bool,
    manifests_dir,
    split_dir,
) -> dict[str, object]:
    """Audited, scope-checked IRC frames via the v1 truth bridge.

    The bridge enforces ``allow_truth`` and frozen-split membership before any
    path is resolved; this module never resolves quarantined paths itself.
    """
    from pes2ts_core.pesn.offline.data.truth_source import load_scoped_irc_frames

    return load_scoped_irc_frames(
        reaction_id,
        allow_truth=allow_truth,
        manifests_dir=manifests_dir,
        split_dir=split_dir,
    )


# ---------------------------------------------------------------------------
# Geometry: TS identification, two-sided split, drift removal
# ---------------------------------------------------------------------------


def ts_position_in_ordered_path(
    *, layout: str, orientation: str, ts_frame_index: int, splice_index: int | None,
) -> int:
    """Index of the TS frame inside the v1 R→TS→P ordered path.

    Mirrors ``direct_preprocess.order_frames_r_to_p``: the ordered path is
    ``prefix + [TS] + suffix`` with ``prefix = branch2[::-1]`` for
    ``P_first`` and ``branch1[::-1]`` for the other orientation.  A
    single-branch layout has no second endpoint side — the caller registers
    the missing branch and skips the reaction (WP1 never fabricates a side).
    """
    if layout != "ts_first_dual_branch":
        raise DualTSPreprocessError(
            "LAYOUT_UNSUPPORTED",
            f"IRC layout {layout!r} is not two-sided; the missing branch "
            "must be registered, not fabricated",
        )
    if splice_index is None:
        raise DualTSPreprocessError(
            "LAYOUT_UNSUPPORTED", "dual-branch layout without splice_index")
    # branch1 = frames[ts+1 : splice+1], branch2 = frames[splice+1 : n]
    return int(splice_index) - int(ts_frame_index)  # prefix length (R_first)


def order_frames_and_energy(
    frames: np.ndarray,
    energies: np.ndarray | None,
    *,
    layout: str,
    orientation: str,
    ts_frame_index: int,
    splice_index: int | None,
) -> tuple[np.ndarray, np.ndarray | None, int]:
    """Order frames (and per-frame energies) R→TS→P via the v1 rule.

    Returns ``(ordered_frames, ordered_energies, ts_position)`` where
    ``ts_position`` is the TS index inside the ordered path.
    """
    from pes2ts_core.pesn.offline.data.direct_preprocess import order_frames_r_to_p

    ordered = order_frames_r_to_p(
        np.asarray(frames, dtype=float),
        layout=layout, orientation=orientation,
        ts_frame_index=int(ts_frame_index),
        splice_index=(None if splice_index is None else int(splice_index)),
    )
    ordered_e: np.ndarray | None = None
    if energies is not None:
        e = np.asarray(energies, dtype=float)
        ts = int(ts_frame_index)
        if layout == "ts_first_dual_branch":
            sp = int(splice_index)  # type: ignore[arg-type]
            branch1_e = e[ts + 1:sp + 1]
            branch2_e = e[sp + 1:]
            ordered_e = (
                np.concatenate([branch2_e[::-1], e[ts:ts + 1], branch1_e])
                if orientation == "P_first"
                else np.concatenate([branch1_e[::-1], e[ts:ts + 1], branch2_e])
            )
        else:
            ordered_e = e
        if ordered_e.shape[0] != ordered.shape[0]:
            raise DualTSPreprocessError(
                "ENERGY_ORDER_MISMATCH",
                f"ordered energy count {ordered_e.shape[0]} != frame count "
                f"{ordered.shape[0]}",
            )
    ts_position = ts_position_in_ordered_path(
        layout=layout, orientation=orientation,
        ts_frame_index=int(ts_frame_index), splice_index=splice_index,
    )
    return ordered, ordered_e, ts_position


def identify_ts_frame(
    energies: np.ndarray | None,
    *,
    authority_ts_position: int | None,
) -> tuple[int, str]:
    """TS identification with the rule used recorded (plan §I.4 step 1).

    The G1 authority ``ts_frame_index`` wins when present; otherwise the
    maximum-energy frame along the combined ordered path is used and the
    fallback rule is returned so it can be recorded in the gaps registry.
    """
    if authority_ts_position is not None:
        return int(authority_ts_position), TS_RULE_AUTHORITY
    if energies is None or len(energies) == 0:
        raise DualTSPreprocessError(
            "TS_FRAME_UNIDENTIFIED",
            "no authority TS index and no per-frame energies available",
        )
    return int(np.argmax(np.asarray(energies, dtype=float))), TS_RULE_MAX_ENERGY


def split_dual_branches(
    ordered_path: np.ndarray, ts_position: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the ordered R→TS→P path into TWO endpoint→TS branches.

    Branch R is ``path[:ts+1]`` (endpoint first, TS last); branch P is
    ``path[:ts+1]`` of the P side reversed, i.e. ``path[ts:][::-1]``.  Each
    branch is internally ordered endpoint→TS; the branches share only the TS
    frame and are never chained across it.
    """
    path = np.asarray(ordered_path, dtype=float)
    k = int(ts_position)
    if not (0 < k < len(path) - 0):
        raise DualTSPreprocessError(
            "TS_POSITION_INVALID",
            f"ts_position {k} outside path of {len(path)} frames",
        )
    branch_r = path[:k + 1]
    branch_p = path[k:][::-1]
    if len(branch_r) < 2 or len(branch_p) < 2:
        raise DualTSPreprocessError(
            "BRANCH_TOO_SHORT",
            f"branches must have >= 2 frames (endpoint + TS); got "
            f"R={len(branch_r)}, P={len(branch_p)}",
        )
    return branch_r, branch_p


def remove_rigid_drift(frames: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """v1's chained rigid drift removal, applied to the COMBINED ordered path.

    Translation + rotation only (mass-weighted Kabsch, det +1).  Registering
    the combined path keeps the TS frame shared identically by both branches;
    per-branch labels are differenced afterwards, so all label identities
    hold by construction and are verified numerically.
    """
    return register_path(np.asarray(frames, dtype=float),
                         np.asarray(masses, dtype=float))


def apply_rigid_transform(
    coords: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray | None = None,
) -> np.ndarray:
    """Apply one rigid transform to coordinates (rotation + translation).

    Vector labels built by differencing transformed coordinates inherit the
    same transform automatically (``(Rx+t) - (Ry+t) = R(x-y)``); the test
    suite verifies this synchronization explicitly.
    """
    r = np.asarray(rotation, dtype=float)
    x = np.asarray(coords, dtype=float)
    if translation is None:
        return x @ r.T
    return x @ r.T + np.asarray(translation, dtype=float)


def rotation_matrix_about_z(theta: float) -> np.ndarray:
    """Deterministic proper rotation (det +1) used by tests/registration checks."""
    c, s = float(np.cos(theta)), float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


# ---------------------------------------------------------------------------
# Endpoint graphs: G1 authority bonds, or deterministic covalent fallback
# ---------------------------------------------------------------------------


def atom_map_ids_for(rows: int) -> tuple[str, ...]:
    """Shared atom mapping: IRC atom row ``i`` ↔ map ``i+1`` (G1 archive
    invariant, ``endpoint_adapter``).  Recorded into every record."""
    return tuple(f"m{i + 1:04d}" for i in range(int(rows)))


def endpoint_graph_from_bond_list(
    atomic_numbers: np.ndarray,
    bonds_map_space: Sequence[Sequence[float]],
) -> EndpointGraph:
    """Endpoint graph from a G1 P1 bond list (map space, 1-based, i<j)."""
    z = np.asarray(atomic_numbers, dtype=int)
    bonds = tuple(
        EndpointBond(i=int(a) - 1, j=int(b) - 1, bond_order=float(o))
        for a, b, o in (tuple(x[:3]) for x in bonds_map_space)
    )
    graph = EndpointGraph(
        atomic_numbers=z,
        atom_map_ids=atom_map_ids_for(int(z.size)),
        bonds=bonds,
    )
    graph.validate()
    return graph


def endpoint_graph_from_geometry(
    atomic_numbers: np.ndarray, coords: np.ndarray,
) -> EndpointGraph:
    """Deterministic covalent-radius single-link perception (fallback).

    Used only when the G1 authority document is unavailable entirely; the
    caller must record :data:`BOND_PERCEPTION_COVALENT` in the task
    ``method_id``/provenance.  Bond order is a single bond (1.0); stereo
    known-masks stay False.
    """
    z = np.asarray(atomic_numbers, dtype=int)
    x = np.asarray(coords, dtype=float)
    n = int(z.size)
    bonds: list[EndpointBond] = []
    for i in range(n):
        ri = COVALENT_RADII.get(int(z[i]))
        if ri is None:
            raise DualTSPreprocessError(
                "COVALENT_RADIUS_MISSING", f"no radius registered for Z={int(z[i])}")
        for j in range(i + 1, n):
            rj = COVALENT_RADII.get(int(z[j]))
            if rj is None:
                raise DualTSPreprocessError(
                    "COVALENT_RADIUS_MISSING",
                    f"no radius registered for Z={int(z[j])}")
            dist = float(np.linalg.norm(x[i] - x[j]))
            if dist <= ri + rj + COVALENT_BOND_TOL:
                bonds.append(EndpointBond(i=i, j=j, bond_order=1.0))
    graph = EndpointGraph(
        atomic_numbers=z,
        atom_map_ids=atom_map_ids_for(n),
        bonds=tuple(bonds),
    )
    graph.validate()
    return graph


def bond_cross_check(
    atomic_numbers: np.ndarray,
    graph_r: EndpointGraph,
    graph_p: EndpointGraph,
    coords_r_endpoint: np.ndarray,
    coords_p_endpoint: np.ndarray,
) -> dict:
    """Weak-match policy (WP2, frozen): authority bonds vs geometry perception.

    Cross-checks the G1 authority ``r_bonds``/``p_bonds`` against the
    deterministic covalent-radius single-link perception (SAME radii and
    tolerance as the fallback, :data:`COVALENT_RADII` +
    :data:`COVALENT_BOND_TOL`) run on the ACTUAL registered endpoint frames.

    Outcomes:

    * ``geometry_verified``  — perception agrees with authority on both ends.
    * ``accept_with_note``   — disagreements on spectator bonds only (bond
      status identical across endpoints per authority): accept with notes.
    * ``disputed``           — disagreement on a CHANGED bond (status differs
      between endpoints): the caller must flag the task
      ``bond_authority=disputed`` + a mapping problem; the reaction still
      stays in the denominator (never dropped).
    * ``crosscheck_unavailable`` — perception impossible (e.g. a covalent
      radius is not registered); recorded honestly, authority is kept.
    """
    z = np.asarray(atomic_numbers, dtype=int)
    base = {
        "perception": BOND_PERCEPTION_COVALENT,
        "radii_source": "cordero_2008_single_bond",
        "tolerance_angstrom": float(COVALENT_BOND_TOL),
        "r_bonds_authority": int(len(graph_r.bonds)),
        "p_bonds_authority": int(len(graph_p.bonds)),
        "disputed_bonds": [],
        "spectator_disagreements": [],
        "notes": [],
        "mapping_problems": [],
    }
    try:
        geo_r = endpoint_graph_from_geometry(z, coords_r_endpoint)
        geo_p = endpoint_graph_from_geometry(z, coords_p_endpoint)
    except DualTSPreprocessError as exc:
        return dict(base, outcome="crosscheck_unavailable",
                    reason=f"geometry perception unavailable: {exc}",
                    notes=[f"bond cross-check skipped: {exc}"])
    auth_r = {(int(b.i), int(b.j)) for b in graph_r.bonds}
    auth_p = {(int(b.i), int(b.j)) for b in graph_p.bonds}
    perceived_r = {(int(b.i), int(b.j)) for b in geo_r.bonds}
    perceived_p = {(int(b.i), int(b.j)) for b in geo_p.bonds}
    changed = auth_r ^ auth_p
    disagreements: set[tuple[str, int, int]] = set()
    disagreements |= {("R", i, j) for (i, j) in (auth_r ^ perceived_r)}
    disagreements |= {("P", i, j) for (i, j) in (auth_p ^ perceived_p)}
    disputed = sorted(t for t in disagreements if (t[1], t[2]) in changed)
    spectators = sorted(t for t in disagreements if (t[1], t[2]) not in changed)
    for side, i, j in disputed:
        base["disputed_bonds"].append({"side": side, "i": int(i), "j": int(j)})
        base["notes"].append(
            f"changed bond ({i},{j}): authority and covalent perception "
            f"disagree on the {side} endpoint frame")
        base["mapping_problems"].append(
            f"bond_authority_disputed: changed bond ({i},{j}) authority != "
            f"covalent-radius perception on {side} endpoint frame")
    for side, i, j in spectators:
        base["spectator_disagreements"].append(
            {"side": side, "i": int(i), "j": int(j)})
        base["notes"].append(
            f"spectator bond ({i},{j}): perception mismatch on the {side} "
            "endpoint frame (accept-with-note)")
    if disputed:
        outcome = "disputed"
    elif spectators:
        outcome = "accept_with_note"
    else:
        outcome = "geometry_verified"
    return dict(base, outcome=outcome)


# ---------------------------------------------------------------------------
# Labels (C1: differenced from the EXPORTED coordinates)
# ---------------------------------------------------------------------------


def export_round(coords: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(coords, dtype=float), EXPORT_COORD_DECIMALS)


def compute_branch_labels(
    exported_coords: np.ndarray,
    masses: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-state labels of one branch from EXPORTED coordinates.

    Returns ``(actions, ts_displacements, remaining_arc_lengths)``:

    * ``a_t = X_{t+1} - X_t`` (``None`` conceptually at the terminal — the
      caller drops the last row's action),
    * ``d_t = X* - X_t`` with ``X*`` the exported TS coordinates (the last
      frame of the branch), exactly zero at the terminal,
    * ``ell_t`` = remaining arc length under the bound metric: the unit-mass
      metric when ``masses is None`` (WP1, bound to
      ``ARC_LENGTH_METRIC_VERSION``), or the frozen C5 diagonal-mass metric
      ``||delta||_M = sqrt(sum_i m_i ||delta_i||^2)`` with the v1
      ``masses_for`` table when ``masses`` is given (WP2, bound to
      :data:`ARC_LENGTH_METRIC_V1_DIAG_MASSES`).

    Identities ``X+a=X_next`` and ``d_t - d_{t+1} = a_t`` hold by
    construction and are verified numerically here (post-drift-removal
    check required by plan §I.4 step 3).
    """
    x = np.asarray(exported_coords, dtype=float)
    if len(x) < 2:
        raise DualTSPreprocessError(
            "BRANCH_TOO_SHORT", "labels need >= 2 frames per branch")
    if masses is not None:
        m = np.asarray(masses, dtype=float)
        if m.shape != (x.shape[1],):
            raise DualTSPreprocessError(
                "MASS_SHAPE_MISMATCH",
                f"masses shape {m.shape} incompatible with {x.shape[1]} atoms")
    actions = x[1:] - x[:-1]                       # (T-1, N, 3)
    ts = x[-1]
    d = ts[None, :, :] - x                          # (T, N, 3)
    if masses is None:
        per_step = np.linalg.norm(actions, axis=(1, 2))  # unit-mass step norms
    else:
        per_step = np.sqrt(
            (m[None, :, None] * actions * actions).sum(axis=(1, 2)))
    ells = np.concatenate([
        np.cumsum(per_step[::-1])[::-1], [0.0]       # remaining arc, 0 at TS
    ])
    # Identity verification (must hold by construction).
    err_a = float(np.max(np.abs(x[:-1] + actions - x[1:])))
    err_d = float(np.max(np.abs(d[:-1] - d[1:] - actions)))
    if err_a > ARC_IDENTITY_TOL or err_d > ARC_IDENTITY_TOL:
        raise DualTSPreprocessError(
            "LABEL_IDENTITY_BROKEN",
            f"post-transform identities violated: |X+a-X'|={err_a:.3e}, "
            f"|d_t-d_t+1-a_t|={err_d:.3e}",
        )
    return actions, d, ells


def flag_a_max_exceedances(
    actions: np.ndarray, a_max_angstrom: float,
) -> dict:
    """Flag per-step per-atom |a| > a_max — count + step indices, NO insertion."""
    per_atom = np.linalg.norm(np.asarray(actions, dtype=float), axis=2)  # (T-1,N)
    step_max = per_atom.max(axis=1)
    over = np.nonzero(step_max > float(a_max_angstrom))[0]
    return {
        "a_max_angstrom": float(a_max_angstrom),
        "count": int(over.size),
        "step_indices": [int(i) for i in over],
        "max_abs_per_atom": float(step_max.max()) if step_max.size else 0.0,
        "inserted": False,
        "note": "wp1_flags_only_node_insertion_is_wp2",
    }


def insert_midpoints_for_exceedance(
    exported_coords: np.ndarray,
    a_max_angstrom: float,
    max_rounds: int = 64,
) -> tuple[np.ndarray, list[dict], list[dict]]:
    """Step-5 node insertion (WP2, C1 discipline).

    When any per-atom ``|a_t|`` exceeds ``a_max``, a linear midpoint node is
    inserted into the offending segment (interpolation on the EXPORTED
    coordinates — never clipping, never dropping a successor).  The scan is
    repeated until every per-atom step is inside the bound (each pass halves
    every offending segment, so termination is guaranteed; a hard round cap
    guards against pathological inputs with a typed refusal).

    Returns ``(new_coords, origins, records)`` where ``origins[i]`` says
    whether final row ``i`` is an original frame (with its original step
    index) or an inserted midpoint, and ``records`` carries one JSON-ready
    entry per inserted node (final index, enclosing original steps, and the
    trigger step's max per-atom displacement).
    """
    x = np.asarray(exported_coords, dtype=float)
    if x.ndim != 3:
        raise DualTSPreprocessError(
            "FRAME_SHAPE_MISMATCH", f"expected (T, N, 3), got {x.shape}")
    origins: list[dict] = [
        {"kind": "original", "original_step_index": i} for i in range(len(x))]
    bound = float(a_max_angstrom)
    for _ in range(int(max_rounds)):
        step_max = np.linalg.norm(x[1:] - x[:-1], axis=2).max(axis=1) \
            if len(x) >= 2 else np.zeros(0)
        if not bool((step_max > bound).any()):
            break
        out_x: list[np.ndarray] = [x[0]]
        out_o: list[dict] = [origins[0]]
        for i in range(1, len(x)):
            prev = out_x[-1]
            worst = float(np.linalg.norm(x[i] - prev, axis=1).max())
            if worst > bound:
                out_x.append(0.5 * (x[i] + prev))
                out_o.append({
                    "kind": "inserted",
                    "original_step_index": None,
                    "trigger_max_abs_per_atom_angstrom": worst,
                })
            out_x.append(x[i])
            out_o.append(origins[i])
        x = np.asarray(out_x, dtype=float)
        origins = out_o
    else:
        raise DualTSPreprocessError(
            "INSERTION_NOT_CONVERGED",
            f"midpoint insertion did not converge within {max_rounds} rounds",
        )

    def _nearest_original(idx: int, direction: int) -> int:
        j = idx + direction
        while 0 <= j < len(origins):
            if origins[j]["kind"] == "original":
                return int(origins[j]["original_step_index"])
            j += direction
        return -1

    records: list[dict] = []
    for idx, origin in enumerate(origins):
        if origin["kind"] != "inserted":
            continue
        records.append({
            "inserted_node_index": int(idx),
            "between_original_steps": [
                _nearest_original(idx, -1), _nearest_original(idx, +1)],
            "trigger_max_abs_per_atom_angstrom":
                float(origin["trigger_max_abs_per_atom_angstrom"]),
            "kind": "midpoint_linear",
        })
    return x, origins, records


# ---------------------------------------------------------------------------
# Task↔teacher cross-record identity (WP2 checklist item, done now)
# ---------------------------------------------------------------------------


def assert_task_teacher_identity(
    task: DualEndpointTask, teacher: DualTSTeacher,
) -> None:
    """Endpoint atomic numbers / mapping vs teacher-state identity.

    Every teacher state must carry exactly the shared atom order of BOTH
    task endpoint graphs; violations are typed rejections, never warnings.
    """
    z = [np.asarray(e.atomic_numbers, dtype=int) for e in task.endpoints]
    maps = [tuple(e.atom_map_ids) for e in task.endpoints]
    for state in teacher.states:
        sz = np.asarray(state.atomic_numbers, dtype=int)
        for side in (0, 1):
            if sz.shape != z[side].shape or not np.array_equal(sz, z[side]):
                raise DualTSPreprocessError(
                    "TASK_TEACHER_IDENTITY_MISMATCH",
                    f"state {state.state_id} atomic numbers disagree with "
                    f"endpoint {['A', 'B'][side]}",
                )
            if tuple(state.atom_map_ids) != maps[side]:
                raise DualTSPreprocessError(
                    "TASK_TEACHER_IDENTITY_MISMATCH",
                    f"state {state.state_id} atom mapping disagrees with "
                    f"endpoint {['A', 'B'][side]}",
                )
    if teacher.reaction_group_id != task.reaction_group_id:
        raise DualTSPreprocessError(
            "TASK_TEACHER_IDENTITY_MISMATCH",
            "task and teacher reference different reaction groups",
        )


# ---------------------------------------------------------------------------
# One-reaction WP1 build
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DualTSPrepOutcome:
    """Per-reaction outcome: records + itemized gaps (honest small package)."""

    reaction_id: str
    reaction_group_id: str
    partition: str
    status: str                       # "ok" | "excluded"
    reason: str
    task: DualEndpointTask | None = None
    teacher: DualTSTeacher | None = None
    task_payload: dict | None = None
    teacher_payload: dict | None = None
    gaps: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


def build_dual_ts_reaction(
    *,
    reaction_id: str,
    reaction_group_id: str,
    partition: str,
    atomic_numbers: Sequence[int],
    ordered_path: np.ndarray,
    authority_ts_position: int | None,
    energies: np.ndarray | None,
    graph_r: EndpointGraph,
    graph_p: EndpointGraph,
    bond_perception: str = BOND_PERCEPTION_AUTHORITY,
    method_id: str = "B3LYPD3-TZVP",
    source: str = "dual-branch DFT IRC via audited bridge (zenodo-18551029-rev1)",
    charge: int = 0,
    multiplicity: int = 1,
    electronic_state: str = "S0",
    a_max_angstrom: float = A_MAX_ANGSTROM,
    arc_length_metric_version: str = ARC_LENGTH_METRIC_VERSION,
    versions: Mapping[str, str] | None = None,
    quality_notes: Sequence[str] = (),
    observation_missing_reason: str = DEFAULT_OBS_MISSING_REASON,
    ts_observation_missing_reason: str = DEFAULT_TS_OBS_MISSING_REASON,
    ell_masses: np.ndarray | None = None,
    insert_on_exceedance: bool = False,
    crosscheck_endpoint_bonds: bool = False,
) -> DualTSPrepOutcome:
    """Full WP1 build for one reaction from plain values (no truth access here).

    ``ordered_path`` is the combined ordered path (endpoint R → TS → endpoint
    P); ``authority_ts_position`` is the TS index inside that path (or None
    to fall back to the maximum-energy frame, which is then recorded).

    WP2 additions (all opt-in; WP1 behavior unchanged when omitted):

    * ``ell_masses`` + ``arc_length_metric_version=
      ARC_LENGTH_METRIC_V1_DIAG_MASSES`` — compute ``ell`` under the frozen
      C5 diagonal-mass metric (masses from the same v1 ``masses_for`` table
      used for registration).  The metric version and the masses must be
      given together: placeholder ⟺ unit-mass, diag-mass string ⟺ masses.
    * ``insert_on_exceedance`` — step-5 midpoint insertion on any per-atom
      ``|a| > a_max`` (C1: labels re-differenced from the exported coords,
      identities re-asserted, ``inserted`` flags + per-insert gap entries).
    * ``crosscheck_endpoint_bonds`` — weak-match policy: authority endpoint
      bonds vs covalent-radius perception on the actual endpoint frames;
      changed-bond disputes flag ``bond_authority=disputed`` +
      ``mapping_problems`` (the reaction still stays in the denominator).
    """
    versions = dict(versions or {})
    z = np.asarray(atomic_numbers, dtype=int)
    path = np.asarray(ordered_path, dtype=float)
    if path.ndim != 3 or path.shape[1:] != (int(z.size), 3):
        raise DualTSPreprocessError(
            "FRAME_SHAPE_MISMATCH",
            f"path shape {path.shape} incompatible with {z.size} atoms",
        )
    masses = masses_for(z)  # v1 element/mass table (typed refusal on unknown Z)
    # C5 binding gate: the metric version string must match the metric that
    # actually produced ell (placeholder ⟺ unit-mass; diag-mass ⟺ masses).
    metric_is_mass = str(arc_length_metric_version) == \
        ARC_LENGTH_METRIC_V1_DIAG_MASSES
    if metric_is_mass != (ell_masses is not None):
        raise DualTSPreprocessError(
            "METRIC_VERSION_MISMATCH",
            "arc_length_metric_version must be "
            f"{ARC_LENGTH_METRIC_V1_DIAG_MASSES!r} exactly when ell_masses is "
            f"given (got masses={ell_masses is not None}, "
            f"version={arc_length_metric_version!r})")
    ell_m = masses if metric_is_mass else None

    # 1. TS identification (rule recorded).
    ts_position, ts_rule = identify_ts_frame(
        energies, authority_ts_position=authority_ts_position)
    if not (0 < ts_position < len(path)):
        raise DualTSPreprocessError(
            "TS_POSITION_INVALID",
            f"identified TS position {ts_position} outside 1..{len(path) - 1}",
        )

    # 3. Rigid drift removal on the combined path (shared TS frame).
    registered = remove_rigid_drift(path, masses)

    # 2. Two-sided split, each branch internally endpoint→TS, no cross-TS.
    branch_r_frames, branch_p_frames = split_dual_branches(registered, ts_position)

    # Endpoint A/B assignment: canonical (content-digest) order of the pair;
    # branch A is the side whose endpoint graph sorts first.  The swap is
    # verified digest-identical below.
    ordered_pair = canonical_endpoint_order(graph_r, graph_p)
    label_of_graph = {id(ordered_pair[0]): "A", id(ordered_pair[1]): "B"}
    branch_of_label = {
        "A": branch_r_frames if label_of_graph[id(graph_r)] == "A"
        else branch_p_frames,
        "B": branch_p_frames if label_of_graph[id(graph_r)] == "A"
        else branch_r_frames,
    }

    # 4/6 (geometry part): export + labels per branch.  WP2: optional
    # step-5 midpoint insertion BEFORE labels (C1: labels are always
    # re-differenced from the final EXPORTED coordinates).
    branch_data: dict[str, dict] = {}
    for label in ("A", "B"):
        frames = export_round(branch_of_label[label])
        inserted_records: list[dict] = []
        if insert_on_exceedance:
            pre = flag_a_max_exceedances(
                frames[1:] - frames[:-1], a_max_angstrom)
            frames_new, _origins, ins_records = insert_midpoints_for_exceedance(
                frames, a_max_angstrom)
            frames = export_round(frames_new)
            post = flag_a_max_exceedances(
                frames[1:] - frames[:-1], a_max_angstrom)
            if post["count"] != 0:
                raise DualTSPreprocessError(
                    "INSERTION_INCOMPLETE",
                    f"branch {label}: {post['count']} steps still exceed "
                    f"a_max={a_max_angstrom} after midpoint insertion")
            exceedance = dict(pre)
            exceedance.update({
                "inserted": True,
                "insertion_count": int(len(ins_records)),
                "post_insertion_count": 0,
                "note": "wp2_midpoint_insertion_applied",
            })
            for rec in ins_records:
                entry = dict(rec)
                entry["branch"] = label
                entry["inserted_state_id"] = \
                    f"{reaction_id}|{label}|{rec['inserted_node_index']:04d}"
                inserted_records.append(entry)
        else:
            exceedance = None
        actions, d, ells = compute_branch_labels(frames, masses=ell_m)
        if exceedance is None:
            exceedance = flag_a_max_exceedances(actions, a_max_angstrom)
        branch_data[label] = {
            "coords": frames,
            "actions": actions,
            "d": d,
            "ells": ells,
            "exceedances": exceedance,
            "inserted_nodes": inserted_records,
        }

    # Weak-match policy (WP2): cross-check authority endpoint bonds against
    # covalent-radius perception on the ACTUAL registered endpoint frames.
    bond_crosscheck_doc: dict | None = None
    bond_authority: str | None = None
    mapping_problems: list[str] = []
    if crosscheck_endpoint_bonds:
        bond_crosscheck_doc = bond_cross_check(
            z, graph_r, graph_p, registered[0], registered[-1])
        bond_authority = {
            "geometry_verified": "geometry_verified",
            "accept_with_note": "accepted_with_note",
            "disputed": "disputed",
            # perception impossible: authority kept, noted honestly
            "crosscheck_unavailable": "accepted_with_note",
        }[bond_crosscheck_doc["outcome"]]
        mapping_problems = list(bond_crosscheck_doc["mapping_problems"])

    # Task record (unordered pair; A/B swap digest identity verified).
    atom_maps = atom_map_ids_for(int(z.size))
    constraints: dict = {
        "a_max_angstrom": float(a_max_angstrom),
        "atom_row_to_map": "row i <-> map i+1 <-> IRC atom row i (g1 invariant)",
        "atom_map_ids": list(atom_maps),
        "endpoint_order": "canonical_endpoint_order (content digest)",
    }
    eligibility = {
        "partition": str(partition),
        "wp": "wp1_small_geometry_only",
        "bond_perception": str(bond_perception),
        "drift_rule": DRIFT_RULE,
        "ts_rule": ts_rule,
    }
    if bond_authority is not None:
        eligibility["bond_authority"] = bond_authority
    if crosscheck_endpoint_bonds:
        eligibility["wp"] = "wp2_fit50_full_teacher"
    task = DualEndpointTask(
        reaction_id=str(reaction_id),
        reaction_group_id=str(reaction_group_id),
        endpoints=(ordered_pair[0], ordered_pair[1]),
        charge=int(charge),
        multiplicity=int(multiplicity),
        constraints=constraints,
        source=str(source),
        method_id=str(method_id),
        eligibility=eligibility,
    )
    task.validate()
    swapped = DualEndpointTask(
        reaction_id=task.reaction_id,
        reaction_group_id=task.reaction_group_id,
        endpoints=(task.endpoints[1], task.endpoints[0]),
        charge=task.charge, multiplicity=multiplicity,
        constraints=constraints, source=task.source,
        method_id=task.method_id, eligibility=eligibility,
    )
    if swapped.content_digest() != task.content_digest():
        raise DualTSPreprocessError(
            "SWAP_DIGEST_MISMATCH",
            "A/B swap changed the task content digest",
        )

    # Teacher record: one branch + one episode per endpoint side.
    episode_versions = {
        "budget_version": versions.get("budget_version",
                                       "wp1_teacher_full_path_v0"),
        "environment_version": versions.get("environment_version",
                                            "wp1_geometry_only_v0"),
        "reward_version": versions.get("reward_version",
                                       "unassigned_wp1_v0"),
    }
    branches: list[BranchRecord] = []
    episodes: list[EpisodeRecord] = []
    states: list[TeacherState] = []
    labels: list[TeacherLabel] = []
    branch_ids: dict[str, str] = {}
    for label in ("A", "B"):
        branch_id = f"{reaction_id}#br{label}"
        branch_ids[label] = branch_id
        episode_id = f"{reaction_id}#ep{label}0"
        data = branch_data[label]
        coords = data["coords"]
        n_steps = len(coords)
        branches.append(BranchRecord(
            branch_id=branch_id, endpoint_label=label,
            reaction_id=str(reaction_id), episode_ids=(episode_id,),
        ))
        episodes.append(EpisodeRecord(
            episode_id=episode_id, branch_id=branch_id,
            reaction_id=str(reaction_id),
            start_state_id=f"{reaction_id}|{label}|0000",
            **episode_versions,
        ))
        for step in range(n_steps):
            is_terminal = step == n_steps - 1
            state_id = f"{reaction_id}|{label}|{step:04d}"
            states.append(TeacherState(
                state_id=state_id,
                episode_id=episode_id,
                branch_id=branch_id,
                step_index=step,
                atomic_numbers=z,
                coordinates=coords[step],
                atom_map_ids=atom_maps,
                method_id=str(method_id),
                electronic_state=str(electronic_state),
                multiplicity=int(multiplicity),
                observation=None,
                observation_missing_reason=(
                    ts_observation_missing_reason if is_terminal
                    else observation_missing_reason),
                boundary_type=(BOUNDARY_TEACHER_ENDPOINT if is_terminal
                               else None),
                next_state_id=(None if is_terminal
                               else f"{reaction_id}|{label}|{step + 1:04d}"),
            ))
            labels.append(TeacherLabel(
                state_id=state_id,
                ts_displacement=data["d"][step],
                remaining_arc_length=float(data["ells"][step]),
                teacher_endpoint=is_terminal,
                action_delta=(None if is_terminal
                              else data["actions"][step]),
            ))

    teacher = DualTSTeacher(
        reaction_id=str(reaction_id),
        reaction_group_id=str(reaction_group_id),
        branches=tuple(branches),
        episodes=tuple(episodes),
        states=tuple(states),
        labels=tuple(labels),
        arc_length_metric_version=str(arc_length_metric_version),
    )
    teacher.validate()

    # WP2 checklist item, executed now: cross-record identity.
    assert_task_teacher_identity(task, teacher)

    geometry_quality = [str(note) for note in quality_notes]
    if energies is not None:
        e_argmax = int(np.argmax(np.asarray(energies, dtype=float)))
        if authority_ts_position is not None and e_argmax != int(authority_ts_position):
            geometry_quality.append(
                f"energy_argmax_frame {e_argmax} != authority ts_frame "
                f"{int(authority_ts_position)} (authority rule kept)")

    gaps = {
        "reaction_id": str(reaction_id),
        "reaction_group_id": str(reaction_group_id),
        "partition": str(partition),
        "status": "ok",
        "reason": "",
        "items": {
            "missing_branches": [],
            "mapping_problems": mapping_problems,
            "geometry_quality_issues": geometry_quality,
            "branch_step_counts": {
                "A": int(len(branch_data["A"]["coords"])),
                "B": int(len(branch_data["B"]["coords"])),
            },
            "a_max_exceedances": {
                "A": branch_data["A"]["exceedances"],
                "B": branch_data["B"]["exceedances"],
            },
            "inserted_nodes": {
                "A": branch_data["A"]["inserted_nodes"],
                "B": branch_data["B"]["inserted_nodes"],
            },
            "bond_cross_check": bond_crosscheck_doc,
            "missing_observations": {
                "A": {
                    "count": int(len(branch_data["A"]["coords"])),
                    "reasons": {
                        "non_ts": observation_missing_reason,
                        "ts": ts_observation_missing_reason,
                    },
                    "xtb": "wp2_scope",
                },
                "B": {
                    "count": int(len(branch_data["B"]["coords"])),
                    "reasons": {
                        "non_ts": observation_missing_reason,
                        "ts": ts_observation_missing_reason,
                    },
                    "xtb": "wp2_scope",
                },
            },
        },
        "ts_rule": ts_rule,
        "drift_rule": DRIFT_RULE,
        "bond_perception": str(bond_perception),
        "arc_length_metric_version": str(arc_length_metric_version),
        "task_digest": task.content_digest(),
        "teacher_digest": teacher.content_digest(),
    }
    diagnostics = {
        "n_frames_ordered": float(len(path)),
        "ts_position": float(ts_position),
        "n_states": float(len(states)),
        "branch_a_steps": float(len(branch_data["A"]["coords"])),
        "branch_b_steps": float(len(branch_data["B"]["coords"])),
        "task_digest": task.content_digest(),
        "teacher_digest": teacher.content_digest(),
        "insertion_count_a": float(len(branch_data["A"]["inserted_nodes"])),
        "insertion_count_b": float(len(branch_data["B"]["inserted_nodes"])),
        "inserted_state_ids": {
            "A": [e["inserted_state_id"]
                  for e in branch_data["A"]["inserted_nodes"]],
            "B": [e["inserted_state_id"]
                  for e in branch_data["B"]["inserted_nodes"]],
        },
        "bond_cross_check_outcome": (
            "" if bond_crosscheck_doc is None
            else str(bond_crosscheck_doc["outcome"])),
    }
    return DualTSPrepOutcome(
        reaction_id=str(reaction_id),
        reaction_group_id=str(reaction_group_id),
        partition=str(partition),
        status="ok",
        reason="",
        task=task,
        teacher=teacher,
        task_payload=task.canonical_payload(),
        teacher_payload=teacher.to_payload(),
        gaps=gaps,
        diagnostics=diagnostics,
    )


def excluded_outcome(
    *,
    reaction_id: str,
    reaction_group_id: str,
    partition: str,
    reason: str,
    error_code: str,
    items: Mapping | None = None,
) -> DualTSPrepOutcome:
    """Honest miss: registered item-by-item, never silently dropped."""
    merged = {
        "missing_branches": [],
        "mapping_problems": [],
        "geometry_quality_issues": [],
        "branch_step_counts": {},
        "a_max_exceedances": {},
        "inserted_nodes": {"A": [], "B": []},
        "bond_cross_check": None,
        "missing_observations": {},
    }
    for key, value in (items or {}).items():
        merged[key] = value
    return DualTSPrepOutcome(
        reaction_id=str(reaction_id),
        reaction_group_id=str(reaction_group_id),
        partition=str(partition),
        status="excluded",
        reason=f"{error_code}: {reason}",
        gaps={
            "reaction_id": str(reaction_id),
            "reaction_group_id": str(reaction_group_id),
            "partition": str(partition),
            "status": "excluded",
            "reason": f"{error_code}: {reason}",
            "error_code": str(error_code),
            "items": merged,
            "task_digest": None,
            "teacher_digest": None,
        },
        diagnostics={"status": 0.0},
    )


__all__ = [
    "A_MAX_ANGSTROM",
    "ARC_IDENTITY_TOL",
    "ARC_LENGTH_METRIC_V1_DIAG_MASSES",
    "BOND_PERCEPTION_AUTHORITY",
    "BOND_PERCEPTION_COVALENT",
    "COVALENT_BOND_TOL",
    "COVALENT_RADII",
    "DEFAULT_OBS_MISSING_REASON",
    "DEFAULT_TS_OBS_MISSING_REASON",
    "DRIFT_RULE",
    "DUAL_TS_PREPROCESS_VERSION",
    "EXPORT_COORD_DECIMALS",
    "DualTSPreprocessError",
    "DualTSPrepOutcome",
    "TS_RULE_AUTHORITY",
    "TS_RULE_MAX_ENERGY",
    "apply_rigid_transform",
    "assert_task_teacher_identity",
    "bond_cross_check",
    "build_dual_ts_reaction",
    "compute_branch_labels",
    "endpoint_graph_from_bond_list",
    "endpoint_graph_from_geometry",
    "excluded_outcome",
    "export_round",
    "flag_a_max_exceedances",
    "identify_ts_frame",
    "insert_midpoints_for_exceedance",
    "load_irc_frames_for_reaction",
    "order_frames_and_energy",
    "remove_rigid_drift",
    "rotation_matrix_about_z",
    "split_dual_branches",
    "ts_position_in_ordered_path",
]
