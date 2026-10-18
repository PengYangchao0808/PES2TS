"""Direct expert-transition preprocessing (WP1, plan v3 §4).

Pipeline per reaction (offline teacher construction; the online environment
shares ``advance_direct_state`` so both produce identical semantics):

1. authority documents (G1 P1 mapping + v2 export) give the task graph
   (row-space edits), atom map, charge/multiplicity and IRC layout metadata;
2. IRC frames are read **through the audited truth bridge** (R1) against the
   frozen direct split;
3. frames are ordered R -> TS -> P from the frozen layout/orientation fields
   and rigid-drift-registered (mass-weighted Kabsch, translation+rotation
   only, no reflection, no per-fragment alignment). Fresh xTB E/g is computed
   at every final node geometry, so no archived gradient is ever rotated in;
4. the registered polyline is resampled by mass-weighted arc length with
   per-atom displacement kept inside ``0.95 * a_max`` (over-limit segments
   split; zero-displacement duplicates removed);
5. each node's E/g comes from ACP (content-addressed receipts; identical
   geometry replays the cache, never recomputes over a conflict);
6. states are advanced through the shared advancer; the episode truncates at
   the first goal terminal, budget exhaustion, QC terminal, or the data-slice
   end (a boundary, never a task terminal).

No piece of this module fabricates observations: a node without a
completed E/g receipt yields a censored/terminal transition exactly as the
online environment would.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Final, Mapping

import numpy as np

from pes2ts_core.generation.planning.direct_mdp import (
    ADVANCER_VERSION,
    DirectObservation,
    advance_direct_state,
    goal_potential,
)
from pes2ts_core.pesn.contracts.direct import (
    DEFAULT_REWARD_SPEC,
    DirectAction,
    DirectRewardSpec,
    DirectState,
    DirectTransition,
)
from pes2ts_core.pesn.offline.data.endpoint_adapter import (
    adapt_reaction,
    load_authority_document,
)

PREPROCESS_VERSION: Final[str] = "pes2ts_direct_preprocess_v1"
RESAMPLE_BOUND_FACTOR: Final[float] = 0.95
MASS_BY_Z: Final[Mapping[int, float]] = {
    1: 1.008, 2: 4.003, 3: 6.94, 4: 9.012, 5: 10.81, 6: 12.011, 7: 14.007,
    8: 15.999, 9: 18.998, 10: 20.18, 11: 22.99, 12: 24.305, 13: 26.982,
    14: 28.085, 15: 30.974, 16: 32.06, 17: 35.45, 18: 39.948, 19: 39.098,
    20: 40.078, 26: 55.845, 27: 58.933, 28: 58.693, 29: 63.546, 30: 65.38,
    34: 78.971, 35: 79.904, 44: 101.07, 45: 102.906, 46: 106.42, 47: 107.868,
    48: 112.414, 53: 126.904, 78: 196.967, 79: 196.967,
}


class DirectPreprocessError(RuntimeError):
    """Typed preprocessing refusal (never silently drops a reaction)."""


def masses_for(atomic_numbers: np.ndarray) -> np.ndarray:
    out = []
    for z in np.asarray(atomic_numbers).astype(int):
        mass = MASS_BY_Z.get(int(z))
        if mass is None:
            raise DirectPreprocessError(f"no mass registered for Z={z}")
        out.append(mass)
    return np.asarray(out, dtype=float)


def kabsch_register(mobile: np.ndarray, reference: np.ndarray,
                    masses: np.ndarray) -> np.ndarray:
    """Mass-weighted rigid registration (translation + rotation, det +1)."""
    w = masses[:, None]
    com_m = (mobile * w).sum(0) / w.sum()
    com_r = (reference * w).sum(0) / w.sum()
    p = mobile - com_m
    q = reference - com_r
    h = (p * w).T @ q
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    correction = np.diag([1.0, 1.0, d])
    rotation = u @ correction @ vt
    return (rotation @ p.T).T + com_r


def register_path(frames: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """Chained registration removing integral rigid drift only."""
    out = np.empty_like(frames, dtype=float)
    out[0] = frames[0]
    for i in range(1, len(frames)):
        out[i] = kabsch_register(frames[i], out[i - 1], masses)
    return out


def order_frames_r_to_p(frames: np.ndarray, *, layout: str, orientation: str,
                        ts_frame_index: int, splice_index: int | None) -> np.ndarray:
    """Freeze the R -> TS -> P frame order from the validated G1 metadata."""
    if layout == "ts_first_dual_branch":
        if splice_index is None:
            raise DirectPreprocessError("dual-branch layout without splice_index")
        branch1 = frames[ts_frame_index + 1:splice_index + 1]
        branch2 = frames[splice_index + 1:]
        ts = frames[ts_frame_index:ts_frame_index + 1]
        if orientation == "P_first":       # branch1 -> P, branch2 -> R
            path = np.concatenate([branch2[::-1], ts, branch1], axis=0)
        else:                              # branch1 -> R, branch2 -> P
            path = np.concatenate([branch1[::-1], ts, branch2], axis=0)
        return path
    if layout == "single_branch":
        branch1 = frames[ts_frame_index + 1:]
        ts = frames[ts_frame_index:ts_frame_index + 1]
        if orientation == "P_first":
            raise DirectPreprocessError(
                "single_branch P-side has no R-side reset; excluded in v1"
            )
        return np.concatenate([branch1[::-1], ts], axis=0)
    raise DirectPreprocessError(f"unsupported IRC layout {layout!r}")


def resample_path(nodes: np.ndarray, masses: np.ndarray, *,
                   action_max: float) -> np.ndarray:
    """Arc-uniform resampling with the per-atom support bound enforced.

    First pass samples uniformly in mass-weighted arc length; any segment
    whose per-atom displacement exceeds ``0.95*a_max`` is split by repeated
    halving (plan §4: insert nodes, never clip actions). Near-zero
    displacement duplicates are dropped.
    """
    bound = RESAMPLE_BOUND_FACTOR * action_max
    diffs = nodes[1:] - nodes[:-1]
    seg = np.sqrt((masses[:, None] * diffs * diffs).sum((1, 2)))
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(arc[-1])
    if total <= 0.0:
        return nodes[:1]

    step = max(bound * float(np.sqrt(masses.min())), 1e-3)
    n_target = max(2, int(np.ceil(total / step)) + 1)
    positions = np.linspace(0.0, total, n_target)
    sampled = np.empty((len(positions), nodes.shape[1], nodes.shape[2]), dtype=float)
    for i, s in enumerate(positions):
        k = int(np.searchsorted(arc, s, side="right") - 1)
        k = min(max(k, 0), len(nodes) - 2)
        span = arc[k + 1] - arc[k]
        t = 0.0 if span <= 0 else (s - arc[k]) / span
        sampled[i] = (1.0 - t) * nodes[k] + t * nodes[k + 1]

    # enforce the per-atom bound by splitting over-limit segments
    changed = True
    while changed:
        changed = False
        out = [sampled[0]]
        for i in range(1, len(sampled)):
            delta = sampled[i] - out[-1]
            if float(np.linalg.norm(delta, axis=1).max()) > bound * (1 + 1e-9):
                mid = 0.5 * (sampled[i] + out[-1])
                out.append(mid)
                out.append(sampled[i])
                changed = True
            else:
                out.append(sampled[i])
        sampled = np.asarray(out, dtype=float)

    # drop near-zero displacement duplicates (keeps the reset node)
    keep = [sampled[0]]
    for i in range(1, len(sampled)):
        if float(np.linalg.norm(sampled[i] - keep[-1], axis=1).max()) > 1e-9:
            keep.append(sampled[i])
    return np.asarray(keep, dtype=float)


@dataclass(frozen=True, slots=True)
class DirectPrepResult:
    reaction_id: str
    group_id: str
    subset: str
    status: str                        # completed | excluded | truncated
    reason: str
    transitions: tuple[DirectTransition, ...] = ()
    task_graph: Mapping = field(default_factory=dict)
    atomic_numbers: np.ndarray | None = None
    atom_map_ids: tuple[str, ...] = ()
    diagnostics: Mapping[str, float] = field(default_factory=dict)


def prepare_direct_reaction(
    reaction_id: str,
    *,
    interim_dir,
    manifests_dir,
    split_dir,
    observation_factory: Callable[[np.ndarray, str], DirectObservation],
    reward_spec: DirectRewardSpec = DEFAULT_REWARD_SPEC,
    subset: str,
    group_id: str,
    action_max: float = 0.1,
    shard_size: int = 1000,
    allow_truth: bool = True,
    max_nodes: int | None = None,
) -> DirectPrepResult:
    """Build expert transitions for one reaction (single-episode, serial)."""
    # lazy import: the audited bridge pulls the HDF5 truth stack; training
    # environments import this module for the shared advancer only and must
    # not acquire a truth-reader dependency
    from pes2ts_core.pesn.offline.data.truth_source import load_scoped_irc_frames

    docs = load_authority_document(interim_dir, reaction_id, shard_size=shard_size)
    adapter = adapt_reaction(docs)
    # map space (dense 1..n) -> row space (0..n-1) exactly as to_task_graph
    task_graph_doc = {
        "edits": [
            {"kind": str(e["kind"]), "i": int(e["maps"][0]) - 1,
             "j": int(e["maps"][1]) - 1,
             "reactant_order": float(e.get("r_order") or 0.0),
             "product_order": float(e.get("p_order") or 0.0)}
            for e in adapter["edits"]
        ],
        "p_bonds": [[int(a) - 1, int(b) - 1, float(o)] for a, b, o in adapter["p_bonds"]],
    }
    atomic_numbers = np.asarray(adapter["atomic_numbers_by_map"], dtype=int)
    masses = masses_for(atomic_numbers)
    atom_map_ids = tuple(str(m) for m in adapter["atom_map"])
    charge = int(adapter.get("charge_total_reactants") or 0)
    multiplicity = int(adapter.get("multiplicity_max") or 1)

    frames_doc = load_scoped_irc_frames(
        reaction_id, allow_truth=allow_truth, manifests_dir=manifests_dir,
        split_dir=split_dir,
    )
    frames = np.asarray(frames_doc["coordinates"], dtype=float)
    if frames.shape[1:] != (atomic_numbers.size, 3):
        raise DirectPreprocessError(
            f"{reaction_id}: frame shape {frames.shape} incompatible with task graph"
        )
    try:
        ordered = order_frames_r_to_p(
            frames, layout=str(adapter["layout"]),
            orientation=str(adapter["orientation"]),
            ts_frame_index=int(adapter["ts_frame_index"]),
            splice_index=(None if adapter["splice_index"] is None
                          else int(adapter["splice_index"])),
        )
    except DirectPreprocessError as exc:
        return DirectPrepResult(
            reaction_id=reaction_id, group_id=group_id, subset=subset,
            status="excluded", reason=str(exc), task_graph=task_graph_doc,
            atomic_numbers=atomic_numbers, atom_map_ids=atom_map_ids,
        )
    registered = register_path(ordered, masses)
    resampled = resample_path(registered, masses, action_max=action_max)
    if max_nodes is not None and len(resampled) > max_nodes:
        resampled = resampled[:max_nodes]

    episode_id = f"{reaction_id}-teacher-v1"
    source_key = f"{reaction_id}"
    observe = observation_factory(atomic_numbers, source_key)

    reset_obs = observe(resampled[0], atomic_numbers,
                        charge=charge,
                        multiplicity=multiplicity,
                        evaluation_id=f"{episode_id}-reset")
    if reset_obs.status != "completed":
        return DirectPrepResult(
            reaction_id=reaction_id, group_id=group_id, subset=subset,
            status="excluded", reason=f"reset_observation_{reset_obs.status}",
            task_graph=task_graph_doc, atomic_numbers=atomic_numbers,
            atom_map_ids=atom_map_ids,
        )
    phi0 = goal_potential(resampled[0], atomic_numbers, task_graph_doc)
    state = DirectState(
        atomic_numbers=atomic_numbers, coordinates=resampled[0],
        task_graph=task_graph_doc, atom_map_ids=atom_map_ids,
        charge=charge, multiplicity=multiplicity,
        method_id="GFN2-xTB",
        energy_hartree=float(reset_obs.energy_hartree),
        energy_reference_hartree=float(reset_obs.energy_hartree),
        gradient_hartree_per_angstrom=(
            None if reset_obs.gradient_hartree_per_angstrom is None
            else np.asarray(reset_obs.gradient_hartree_per_angstrom, dtype=float)
        ),
        observation_mask=True,
        steps_remaining=reward_spec.budget_steps,
        event_state={"goal_consecutive_hits": 0.0, "no_progress_count": 0.0,
                     "best_potential": phi0},
    )
    state.validate()

    transitions: list[DirectTransition] = []
    qc_calls = 1
    cache_hits = 1 if reset_obs.cache_reused else 0
    terminal_reason = "data_slice_boundary"
    for k in range(1, len(resampled)):
        prop = resampled[k] - state.coordinates
        action = DirectAction(
            action_max_angstrom=action_max, latent_u=None, normalized_z=None,
            proposed_angstrom=prop, executed=True, rejection_reason=None,
            log_prob_z=None, log_prob_source="expert_not_probabilistic",
        )
        obs = observe(resampled[k], atomic_numbers,
                      charge=charge,
                      multiplicity=multiplicity,
                      evaluation_id=f"{episode_id}-n{k:04d}")
        qc_calls += 1
        if obs.cache_reused:
            cache_hits += 1
        transition = advance_direct_state(
            state, action, obs, reward_spec=reward_spec,
            episode_id=episode_id, step_index=k - 1, reaction_id=reaction_id,
            group_id=group_id, subset=subset,
            transition_id=f"{episode_id}-t{k - 1:04d}",
            parent_node_hash=state.content_hash(),
        )
        transitions.append(transition)
        if transition.terminated:
            terminal_reason = transition.termination_reason
            break
        if transition.next_state is None:
            terminal_reason = transition.termination_reason
            break
        state = transition.next_state

    status = "completed" if terminal_reason == "goal_connected_proxy" else "truncated"
    return DirectPrepResult(
        reaction_id=reaction_id, group_id=group_id, subset=subset,
        status=status, reason=terminal_reason, transitions=tuple(transitions),
        task_graph=task_graph_doc, atomic_numbers=atomic_numbers,
        atom_map_ids=atom_map_ids,
        diagnostics={
            "n_frames": float(len(frames)),
            "n_ordered": float(len(ordered)),
            "n_nodes": float(len(resampled)),
            "n_transitions": float(len(transitions)),
            "qc_calls": float(qc_calls),
            "qc_cache_hits": float(cache_hits),
            "return_sum": float(sum(t.reward for t in transitions)),
        },
    )


__all__ = [
    "PREPROCESS_VERSION",
    "DirectPreprocessError",
    "DirectPrepResult",
    "kabsch_register",
    "masses_for",
    "order_frames_r_to_p",
    "prepare_direct_reaction",
    "register_path",
    "resample_path",
]
