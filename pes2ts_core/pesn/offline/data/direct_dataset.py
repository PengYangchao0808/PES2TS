"""Direct transition shards + lazy dataset (WP1, plan v3 §4/§10).

Shard triplets mirror the teacher-dataset convention (states/labels npz +
provenance json + manifest) so proven tooling patterns apply unchanged.
Per-reaction keys carry the full direct transition stream; the task graph
and identity metadata live in the provenance JSON (never duplicated per
transition). The loader is lazy: only the manifest and provenance are read
at construction; npz arrays materialize per reaction on demand with a small
LRU shard cache. Expert action log-probabilities are structurally absent
(``expert_not_probabilistic``), never zero-filled.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterator, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.offline.data.direct_preprocess import (
    PREPROCESS_VERSION,
    DirectPrepResult,
)
from pes2ts_core.utils.hashing import sha256_file
from pes2ts_core.utils.jsonio import read_json, write_json

DATASET_SCHEMA_VERSION: Final[str] = "pes2ts_direct_dataset_v1"
DEFAULT_SHARD_SIZE: Final[int] = 25


class DirectDatasetError(RuntimeError):
    """Typed dataset construction/loading refusal."""


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def write_direct_shards(
    results: Sequence[DirectPrepResult],
    output_dir: str | Path,
    *,
    reward_spec_sha256: str,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> dict:
    """Write shard triplets + manifest; returns the manifest document."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    usable = [r for r in results if r.transitions]
    shard_docs = []
    for start in range(0, len(usable), shard_size):
        chunk = usable[start:start + shard_size]
        tag = f"{start // shard_size:04d}"
        states: dict[str, np.ndarray] = {}
        labels: dict[str, np.ndarray] = {}
        prov_reactions: dict[str, dict] = {}
        n_transitions = 0
        for res in chunk:
            rid = res.reaction_id
            transitions = res.transitions
            n = res.atomic_numbers.shape[0]
            t_count = len(transitions)
            s_coords = np.zeros((t_count, n, 3), dtype=np.float32)
            s_grad = np.zeros((t_count, n, 3), dtype=np.float32)
            s_energy = np.zeros(t_count, dtype=np.float64)
            s_steps = np.zeros(t_count, dtype=np.int32)
            s_event = np.zeros((t_count, 3), dtype=np.float32)
            s_de = np.zeros(t_count, dtype=np.float64)
            a_ang = np.zeros((t_count, n, 3), dtype=np.float32)
            r = np.zeros(t_count, dtype=np.float64)
            term = np.zeros(t_count, dtype=bool)
            boot = np.zeros(t_count, dtype=np.float32)
            td_ok = np.zeros(t_count, dtype=bool)
            reasons = []
            for t_i, tr in enumerate(transitions):
                s_coords[t_i] = tr.state.coordinates
                grad = tr.state.gradient_hartree_per_angstrom
                s_grad[t_i] = 0.0 if grad is None else grad
                s_energy[t_i] = 0.0 if tr.state.energy_hartree is None else tr.state.energy_hartree
                s_steps[t_i] = tr.state.steps_remaining
                s_event[t_i] = (
                    tr.state.event_state.get("goal_consecutive_hits", 0.0),
                    tr.state.event_state.get("no_progress_count", 0.0),
                    tr.state.event_state.get("best_potential", 0.0),
                )
                s_de[t_i] = 0.0 if tr.state.delta_e_hartree() is None else tr.state.delta_e_hartree()
                a = tr.action.executed_angstrom
                a_ang[t_i] = 0.0 if a is None else a
                r[t_i] = tr.reward
                term[t_i] = tr.terminated
                boot[t_i] = tr.bootstrap
                td_ok[t_i] = tr.td_eligible
                reasons.append(tr.termination_reason)
            states[f"coords__{rid}"] = s_coords
            states[f"grad__{rid}"] = s_grad
            states[f"z__{rid}"] = res.atomic_numbers.astype(np.int16)
            labels[f"energy__{rid}"] = s_energy
            labels[f"de__{rid}"] = s_de
            labels[f"steps__{rid}"] = s_steps
            labels[f"event__{rid}"] = s_event
            labels[f"action__{rid}"] = a_ang
            labels[f"reward__{rid}"] = r
            labels[f"terminated__{rid}"] = term
            labels[f"bootstrap__{rid}"] = boot
            labels[f"td_eligible__{rid}"] = td_ok
            prov_reactions[rid] = {
                "group_id": res.group_id,
                "subset": res.subset,
                "episode_id": transitions[0].episode_id if transitions else "",
                "status": res.status,
                "reason": res.reason,
                "n_transitions": t_count,
                "termination_reasons": reasons,
                "task_graph": dict(res.task_graph),
                "atom_map_ids": list(res.atom_map_ids),
                "charge": int(transitions[0].state.charge) if transitions else 0,
                "multiplicity": int(transitions[0].state.multiplicity) if transitions else 1,
                "diagnostics": dict(res.diagnostics),
            }
            n_transitions += t_count
        np.savez_compressed(out / f"states_{tag}.npz", **states)
        np.savez_compressed(out / f"labels_{tag}.npz", **labels)
        write_json(out / f"provenance_{tag}.json", {"reactions": prov_reactions})
        shard_docs.append({
            "tag": tag,
            "n_reactions": len(chunk),
            "n_transitions": n_transitions,
            "states_sha256": sha256_file(out / f"states_{tag}.npz"),
            "labels_sha256": sha256_file(out / f"labels_{tag}.npz"),
            "provenance_sha256": sha256_file(out / f"provenance_{tag}.json"),
        })
    excluded = [
        {"reaction_id": r.reaction_id, "group_id": r.group_id, "subset": r.subset,
         "reason": r.reason}
        for r in results if not r.transitions
    ]
    manifest = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "preprocess_version": PREPROCESS_VERSION,
        "reward_spec_sha256": reward_spec_sha256,
        "n_reactions": len(usable),
        "n_transitions": sum(s["n_transitions"] for s in shard_docs),
        "n_excluded": len(excluded),
        "excluded": excluded,
        "shards": shard_docs,
        "units": {"coordinates": "angstrom", "energy": "hartree",
                  "gradient": "hartree/angstrom", "action": "angstrom"},
        "note": "expert actions carry no log-probability by construction",
    }
    write_json(out / "dataset_manifest.json", manifest)
    return manifest


# ---------------------------------------------------------------------------
# Lazy loading
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class DirectSample:
    """One training-ready transition (numpy view, task graph by reference)."""

    reaction_id: str
    group_id: str
    subset: str
    step_index: int
    atomic_numbers: np.ndarray
    coords: np.ndarray                 # (N,3)
    gradient: np.ndarray               # (N,3)
    energy: float
    delta_e: float
    steps_remaining: int
    event_state: np.ndarray            # (3,)
    action: np.ndarray                 # (N,3) executed angstrom
    reward: float
    terminated: bool
    bootstrap: float
    td_eligible: bool
    termination_reason: str
    task_graph: Mapping
    charge: int
    multiplicity: int


class DirectTransitionDataset:
    """Lazy per-reaction view over direct transition shards."""

    def __init__(self, dataset_dir: str | Path, *,
                 shard_cache_size: int = 2,
                 allowed_subsets: Sequence[str] | None = None) -> None:
        self.root = Path(dataset_dir)
        manifest_path = self.root / "dataset_manifest.json"
        if not manifest_path.is_file():
            raise DirectDatasetError(f"direct dataset manifest missing: {manifest_path}")
        self.manifest = read_json(manifest_path)
        if self.manifest.get("schema_version") != DATASET_SCHEMA_VERSION:
            raise DirectDatasetError("direct dataset schema mismatch")
        self._shard_cache: OrderedDict[str, tuple] = OrderedDict()
        self._shard_cache_size = shard_cache_size
        self._index: dict[str, tuple[str, dict]] = {}
        self._subsets: dict[str, str] = {}
        allowed = set(allowed_subsets) if allowed_subsets else None
        for shard in self.manifest["shards"]:
            prov = read_json(self.root / f"provenance_{shard['tag']}.json")
            for rid, meta in prov["reactions"].items():
                subset = str(meta["subset"])
                if allowed is not None and subset not in allowed:
                    continue
                self._index[rid] = (str(shard["tag"]), meta)
                self._subsets[rid] = subset

    # -- access ------------------------------------------------------------

    @property
    def reaction_ids(self) -> list[str]:
        return sorted(self._index)

    @property
    def subsets(self) -> dict[str, str]:
        return dict(self._subsets)

    def reaction_ids_for_subset(self, subset: str) -> list[str]:
        return [rid for rid, s in self._subsets.items() if s == subset]

    def _shard(self, tag: str):
        cached = self._shard_cache.pop(tag, None)
        if cached is not None:
            self._shard_cache[tag] = cached
            return cached
        states = np.load(self.root / f"states_{tag}.npz")
        labels = np.load(self.root / f"labels_{tag}.npz")
        entry = (states, labels)
        self._shard_cache[tag] = entry
        while len(self._shard_cache) > self._shard_cache_size:
            _, evicted = self._shard_cache.popitem(last=False)
            evicted[0].close()
            evicted[1].close()
        return entry

    def get_reaction(self, reaction_id: str) -> list[DirectSample]:
        if reaction_id not in self._index:
            raise DirectDatasetError(f"reaction {reaction_id!r} not in dataset")
        tag, meta = self._index[reaction_id]
        states, labels = self._shard(tag)
        z = np.asarray(states[f"z__{reaction_id}"]).astype(int)
        coords = np.asarray(states[f"coords__{reaction_id}"], dtype=float)
        grad = np.asarray(states[f"grad__{reaction_id}"], dtype=float)
        energy = np.asarray(labels[f"energy__{reaction_id}"], dtype=float)
        de = np.asarray(labels[f"de__{reaction_id}"], dtype=float)
        steps = np.asarray(labels[f"steps__{reaction_id}"])
        event = np.asarray(labels[f"event__{reaction_id}"], dtype=float)
        action = np.asarray(labels[f"action__{reaction_id}"], dtype=float)
        reward = np.asarray(labels[f"reward__{reaction_id}"], dtype=float)
        term = np.asarray(labels[f"terminated__{reaction_id}"], dtype=bool)
        boot = np.asarray(labels[f"bootstrap__{reaction_id}"], dtype=float)
        td_ok = np.asarray(labels[f"td_eligible__{reaction_id}"], dtype=bool)
        reasons = list(meta["termination_reasons"])
        samples = []
        for i in range(coords.shape[0]):
            samples.append(DirectSample(
                reaction_id=reaction_id, group_id=str(meta["group_id"]),
                subset=str(meta["subset"]), step_index=i,
                atomic_numbers=z, coords=coords[i], gradient=grad[i],
                energy=float(energy[i]), delta_e=float(de[i]),
                steps_remaining=int(steps[i]), event_state=event[i],
                action=action[i], reward=float(reward[i]),
                terminated=bool(term[i]), bootstrap=float(boot[i]),
                td_eligible=bool(td_ok[i]),
                termination_reason=str(reasons[i]) if i < len(reasons) else "",
                task_graph=meta["task_graph"], charge=int(meta["charge"]),
                multiplicity=int(meta["multiplicity"]),
            ))
        return samples

    def iter_samples(self) -> Iterator[DirectSample]:
        for rid in self.reaction_ids:
            yield from self.get_reaction(rid)

    @property
    def open_shard_count(self) -> int:
        return len(self._shard_cache)


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "DirectDatasetError",
    "DirectSample",
    "DirectTransitionDataset",
    "write_direct_shards",
]
