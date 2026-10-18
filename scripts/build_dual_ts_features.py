#!/usr/bin/env python3
"""Real dual-TS feature build (WP3-A, plan v4 §D WP3 row).

Loads the WP2 fit50 observed teacher table (50 reactions / 100 branches /
6067 states, every state bound to a status=ok xTB observation), freezes the
``pesn_dual_ts_features_v3`` spec (s_g / s_ΔE estimated from the FULL fit
population per the pre-registered rules in
``pes2ts_core.pesn.features.dual_ts_spec``; v4.1 WP1 = observation-only
5-channel node scalars), and builds the feature shards
for ALL states with the disputed-5 bond rule applied.

``--spec`` (additive, used by the WP4 monitor8 build): consume a PRE-FROZEN
``features_spec.json`` (e.g. the wp3 fit spec) instead of re-estimating the
scales — the manifest then records ``spec_provenance = "reused_frozen"``
(NOT re-measured on the new population; C5/G4 isolation condition) plus the
source path, and the scaled-median sanity value is recorded as an honest
observation instead of the fit-population ≈1.0 assertion.

Outputs (default under outputs/pesn/dual_ts_v2_20261015/wp3_features_v41/):

* ``features_spec.json``          — the frozen, content-digested spec
* ``features_RXN_*.npz``          — one shard per reaction
* ``features_manifest.json``      — spec digest, per-reaction counts,
  norm params, bond_source flags, input dims, sanity stats, wall time

No training, no ACP/xTB execution, no network: features are assembled from
the bound observations already on disk.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.contracts.dual_ts import (  # noqa: E402
    DualEndpointTask,
    DualTSTeacher,
    content_digest_of,
)
from pes2ts_core.pesn.features.dual_ts_graph import (  # noqa: E402
    build_state_features,
    build_teacher_supervision,
    build_topology_features,
)
from pes2ts_core.pesn.features.dual_ts_spec import (  # noqa: E402
    build_features_spec,
    estimate_fit_scales_from_records,
    load_spec_payload,
    save_spec_payload,
)
from pes2ts_core.pesn.offline.data.dual_ts_preprocess import (  # noqa: E402
    assert_task_teacher_identity,
)

MANIFEST_SCHEMA: str = "pes2ts_dual_ts_wp41_features_manifest_v1"
TOOL_VERSION: str = "pes2ts_dual_ts_features_build_wp41_observation_only_v1"

DEFAULT_OBSERVED_DIR: str = "outputs/pesn/dual_ts_v2_20261015/wp2_fit50_observed"
DEFAULT_OUT_DIR: str = "outputs/pesn/dual_ts_v2_20261015/wp3_features_v41"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def load_reaction_pair(
    task_path: Path, teacher_path: Path,
) -> tuple[DualEndpointTask, DualTSTeacher]:
    task = DualEndpointTask.from_payload(
        json.loads(task_path.read_text(encoding="utf-8")))
    teacher = DualTSTeacher.from_payload(
        json.loads(teacher_path.read_text(encoding="utf-8")))
    # WP2 checklist item re-run at consume time: endpoint Z/mapping must
    # match every teacher state exactly.
    assert_task_teacher_identity(task, teacher)
    return task, teacher


def episode_start_energy(teacher: DualTSTeacher) -> dict[str, float]:
    """E of each episode's start (endpoint) state from the SAME table."""
    by_id = {s.state_id: s for s in teacher.states}
    out: dict[str, float] = {}
    for episode in teacher.episodes:
        state = by_id.get(episode.start_state_id)
        if state is None or state.observation is None:
            raise RuntimeError(
                f"EPISODE_START_UNBOUND: {teacher.reaction_id} episode "
                f"{episode.episode_id} start {episode.start_state_id}")
        if state.observation.status != "ok":
            raise RuntimeError(
                f"OBS_NOT_OK: start state {state.state_id} status "
                f"{state.observation.status!r}")
        out[episode.episode_id] = float(state.observation.energy_hartree)
    return out


def scale_records_for_reaction(
    teacher: DualTSTeacher,
) -> list[dict[str, Any]]:
    """One record per state: (gradient, delta_e) with ΔE bound to the
    episode start read from the SAME observation table (no recomputation)."""
    e_start = episode_start_energy(teacher)
    records: list[dict[str, Any]] = []
    for state in teacher.states:
        obs = state.observation
        if obs is None or obs.status != "ok":
            raise RuntimeError(
                f"OBS_NOT_OK: {state.state_id} status "
                f"{getattr(obs, 'status', None)!r} (wp2_fit50 must be fully "
                "bound before feature build)")
        g = np.asarray(obs.gradient_hartree_per_angstrom, dtype=float)
        if g.shape != (int(np.asarray(state.atomic_numbers).size), 3):
            raise RuntimeError(
                f"GRAD_SHAPE_INVALID: {state.state_id} gradient {g.shape}")
        delta_e = float(obs.energy_hartree) - e_start[state.episode_id]
        records.append({"gradient": g, "delta_e": delta_e})
    return records


def percentile_summary(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    return {
        "n": int(v.size),
        "min": float(v.min()),
        "p05": float(np.percentile(v, 5)),
        "median": float(np.median(v)),
        "mean": float(v.mean()),
        "p95": float(np.percentile(v, 95)),
        "max": float(v.max()),
    }


def build_reaction_shard(
    task: DualEndpointTask,
    teacher: DualTSTeacher,
    *,
    s_g: float,
    s_delta_e: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Feature shard arrays + per-reaction meta for ONE reaction."""
    topo = build_topology_features(task=task, teacher=teacher)
    n = topo.n_atoms
    by_id = {s.state_id: s for s in teacher.states}
    label_by_state = {lab.state_id: lab for lab in teacher.labels}
    e_start = episode_start_energy(teacher)

    rows: list[dict[str, np.ndarray]] = []
    for state in teacher.states:
        obs = state.observation
        if obs is None or obs.status != "ok":
            raise RuntimeError(f"OBS_NOT_OK: {state.state_id}")
        is_terminal = state.boundary_type == "teacher_endpoint"
        label = label_by_state[state.state_id]
        feats = build_state_features(
            state_id=state.state_id,
            atomic_numbers=state.atomic_numbers,
            coordinates=state.coordinates,
            gradient=obs.gradient_hartree_per_angstrom,
            delta_e_hartree=(
                float(obs.energy_hartree) - e_start[state.episode_id]),
            s_g=s_g,
            s_delta_e=s_delta_e,
            charge=int(task.charge),
            multiplicity=int(task.multiplicity),
        )
        supervision = build_teacher_supervision(
            action_delta=label.action_delta,
            is_teacher_endpoint=is_terminal,
            n_atoms=n,
        )
        if feats.atomic_numbers.size != n:
            raise RuntimeError(f"ATOM_COUNT_MISMATCH: {state.state_id}")
        rows.append({
            "state_id": feats.state_id,
            "branch_id": state.branch_id,
            "episode_id": state.episode_id,
            "step_index": np.int64(state.step_index),
            "is_teacher_endpoint": supervision.is_teacher_endpoint,
            "next_state_id": (state.next_state_id or ""),
            "node_scalars": feats.node_scalars,
            "x_angstrom": feats.x_centered,
            "g_scaled": feats.g_scaled,
            "delta_e_scaled": np.float64(feats.delta_e_scaled),
            "delta_e_hartree": np.float64(
                float(obs.energy_hartree) - e_start[state.episode_id]),
            "near_zero_action_mask": supervision.near_zero_action_mask,
        })

    arrays: dict[str, np.ndarray] = {
        "state_id": np.asarray([r["state_id"] for r in rows], dtype=np.str_),
        "branch_id": np.asarray([r["branch_id"] for r in rows], dtype=np.str_),
        "episode_id": np.asarray([r["episode_id"] for r in rows], dtype=np.str_),
        "step_index": np.asarray([r["step_index"] for r in rows], dtype=np.int64),
        "is_teacher_endpoint": np.asarray(
            [r["is_teacher_endpoint"] for r in rows], dtype=bool),
        "next_state_id": np.asarray(
            [r["next_state_id"] for r in rows], dtype=np.str_),
        "atomic_numbers": topo.atomic_numbers.astype(np.int64),
        "node_scalars": np.stack([r["node_scalars"] for r in rows]).astype(np.float64),
        "x_angstrom": np.stack([r["x_angstrom"] for r in rows]).astype(np.float64),
        "g_scaled": np.stack([r["g_scaled"] for r in rows]).astype(np.float64),
        "delta_e_scaled": np.asarray(
            [r["delta_e_scaled"] for r in rows], dtype=np.float64),
        "delta_e_hartree": np.asarray(
            [r["delta_e_hartree"] for r in rows], dtype=np.float64),
        "near_zero_action_mask": np.stack(
            [r["near_zero_action_mask"] for r in rows]).astype(bool),
        "edge_index": topo.edge_index.astype(np.int64),
        "edge_pairs": topo.edge_pairs.astype(np.int64),
        "atom_feats_a": topo.atom_feats_a.astype(np.float32),
        "atom_feats_b": topo.atom_feats_b.astype(np.float32),
        "edge_feats_a": topo.edge_feats_a.astype(np.float32),
        "edge_feats_b": topo.edge_feats_b.astype(np.float32),
        "bond_source_a": np.asarray(topo.bond_source_a, dtype=np.str_),
        "bond_source_b": np.asarray(topo.bond_source_b, dtype=np.str_),
        "reaction_id": np.asarray(task.reaction_id, dtype=np.str_),
        "reaction_group_id": np.asarray(task.reaction_group_id, dtype=np.str_),
    }
    # Finite checks on every float block before anything is written.
    for key in ("node_scalars", "x_angstrom", "g_scaled",
                "atom_feats_a", "atom_feats_b",
                "edge_feats_a", "edge_feats_b"):
        if not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"NONFINITE_SHARD: {task.reaction_id} {key}")
    meta = {
        "reaction_id": task.reaction_id,
        "reaction_group_id": task.reaction_group_id,
        "states": int(len(rows)),
        "n_atoms": int(n),
        "n_edges_union": topo.n_edges_undirected,
        "bond_source_a": topo.bond_source_a,
        "bond_source_b": topo.bond_source_b,
        "bond_authority_flag": str(task.eligibility.get("bond_authority", "")),
        "disputed": (
            topo.bond_source_a == "covalent_fallback"
            or topo.bond_source_b == "covalent_fallback"),
    }
    return arrays, meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--observed-dir", default=DEFAULT_OBSERVED_DIR,
                        help="WP2 observed teacher table directory")
    parser.add_argument("--out", default=DEFAULT_OUT_DIR,
                        help="output directory for shards + manifest")
    parser.add_argument("--spec", default=None,
                        help="optional pre-frozen features_spec.json to "
                             "consume instead of re-estimating scales")
    parser.add_argument("--wave-label", default="fit",
                        help="population wave label recorded in the manifest "
                             "(default 'fit', unchanged; the WP4 monitor "
                             "build passes 'monitor')")
    args = parser.parse_args(argv)

    wall_start = time.perf_counter()
    observed_dir = Path(args.observed_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    task_paths = sorted(glob.glob(str(observed_dir / "RXN_*.task.json")))
    if not task_paths:
        raise SystemExit(f"no RXN_*.task.json under {observed_dir}")
    reactions: list[tuple[DualEndpointTask, DualTSTeacher]] = []
    for task_path in task_paths:
        teacher_path = Path(str(task_path).replace(".task.json", ".teacher.json"))
        if not teacher_path.exists():
            raise SystemExit(f"missing teacher payload for {task_path}")
        reactions.append(load_reaction_pair(Path(task_path), teacher_path))
    reactions.sort(key=lambda pair: pair[0].reaction_id)

    # ---- Pass 1: fit-population scale estimation (frozen rules) ----------
    all_records: list[dict[str, Any]] = []
    vocab: set[int] = set()
    atom_counts: list[int] = []
    for task, teacher in reactions:
        all_records.extend(scale_records_for_reaction(teacher))
        vocab.update(int(z) for z in task.endpoints[0].atomic_numbers)
        atom_counts.append(int(np.asarray(task.endpoints[0].atomic_numbers).size))
    scales = estimate_fit_scales_from_records(all_records)
    population_states = len(all_records)

    # ---- Frozen spec: build (or load) + save ------------------------------
    n_atoms_range = (int(min(atom_counts)), int(max(atom_counts)))
    if args.spec:
        spec, payload = load_spec_payload(args.spec)
        s_g = float(spec.frozen_scales["s_g"])
        s_delta_e = float(spec.frozen_scales["s_delta_e"])
        spec_provenance = "reused_frozen"
        spec_source_path = str(args.spec)
    else:
        population = {
            "wave": "fit",
            "reactions": len(reactions),
            "branches": sum(len(t.branches) for _, t in reactions),
            "states": population_states,
            "inserted_included": True,
            "ts_terminal_included": True,
        }
        spec, payload = build_features_spec(
            s_g=scales["s_g"],
            s_delta_e=scales["s_delta_e"],
            element_vocab=sorted(vocab),
            population=population,
            n_atoms_range=n_atoms_range,
        )
        s_g = scales["s_g"]
        s_delta_e = scales["s_delta_e"]
        spec_provenance = "estimated_from_fit_population_this_run"
        spec_source_path = None
    spec_path = out_dir / "features_spec.json"
    spec_digest = save_spec_payload(payload, spec_path)

    # ---- Pass 2: per-reaction feature shards ------------------------------
    per_reaction: list[dict[str, Any]] = []
    gtilde_norms: list[np.ndarray] = []
    delta_e_scaled_all: list[float] = []
    n_terminal = 0
    for task, teacher in reactions:
        arrays, meta = build_reaction_shard(
            task, teacher, s_g=s_g, s_delta_e=s_delta_e)
        n_terminal += int(arrays["is_teacher_endpoint"].sum())
        gtilde_norms.append(np.linalg.norm(arrays["g_scaled"], axis=2).ravel())
        delta_e_scaled_all.extend(arrays["delta_e_scaled"].tolist())
        shard_path = out_dir / f"features_{task.reaction_id}.npz"
        np.savez_compressed(shard_path, **arrays)
        meta["npz_file"] = shard_path.name
        meta["npz_sha256"] = sha256_file(shard_path)
        per_reaction.append(meta)

    # ---- Sanity stats + manifest ------------------------------------------
    gtilde = np.concatenate(gtilde_norms)
    # With the estimated-from-this-population spec, s_g IS the population
    # median of per-atom norms, so the scaled median must be 1.0 to floating
    # accuracy.  With a REUSED frozen spec (monitor build) the deviation is
    # expected and honest — it is recorded, never enforced (no re-scaling).
    median_check = float(np.median(gtilde))
    if spec_provenance == "estimated_from_fit_population_this_run" \
            and abs(median_check - 1.0) > 1e-9:
        raise RuntimeError(
            f"SANITY_FAIL: median |g̃| = {median_check!r} != 1 after scaling")
    bond_source_counts: Counter[str] = Counter()
    for meta in per_reaction:
        bond_source_counts[f"a:{meta['bond_source_a']}"] += 1
        bond_source_counts[f"b:{meta['bond_source_b']}"] += 1

    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "tool_version": TOOL_VERSION,
        "study": "dual_ts_v2 WP3-A v4.1 (frozen features_v3 observation-only "
                 "spec + dual-graph exchange-symmetric feature build, no "
                 "training)",
        "observed_dir": str(observed_dir),
        "out_dir": str(out_dir),
        "wall_seconds": round(time.perf_counter() - wall_start, 3),
        "features_spec_file": spec_path.name,
        "spec_digest": spec_digest,
        "spec_payload": payload,
        "spec_provenance": spec_provenance,
        "spec_source_path": spec_source_path,
        "spec_reuse_note": (
            "scales REUSED from the frozen fit spec (never re-measured on "
            "this population)" if spec_provenance == "reused_frozen"
            else "scales estimated from this build's full population"),
        "gtilde_median_check_enforced": (
            spec_provenance == "estimated_from_fit_population_this_run"),
        "population": {
            "wave": str(args.wave_label),
            "reactions": len(reactions),
            "branches": sum(len(t.branches) for _, t in reactions),
            "states": population_states,
            "teacher_endpoint_states": n_terminal,
        },
        "norm_params": {
            "s_g_hartree_per_angstrom": s_g,
            "s_g_rule": payload["support_domain"]["s_g_rule"],
            "s_delta_e_hartree": s_delta_e,
            "s_delta_e_rule": payload["support_domain"]["s_delta_e_rule"],
            "delta_energy_definition": payload["delta_energy_definition"],
        },
        "input_dims": {
            "input_dim": int(payload["input_dim"]),
            "declared_layout": dict(payload["declared_layout"]),
            "equivariant_vectors": list(payload["equivariant_vectors"]),
        },
        "sanity_stats": {
            "gtilde_norm_distribution": percentile_summary(gtilde),
            "gtilde_median_is_one_check": median_check,
            "delta_e_scaled_range": {
                "min": float(min(delta_e_scaled_all)),
                "max": float(max(delta_e_scaled_all)),
            },
            "delta_e_hartree_abs_p95_check": float(np.percentile(
                np.abs(np.asarray(
                    [rec["delta_e"] for rec in all_records], dtype=float)),
                95.0)),
            "atom_counts": {
                "min": n_atoms_range[0],
                "max": n_atoms_range[1],
            },
            "element_vocabulary": sorted(vocab),
        },
        "bond_source_counts": dict(sorted(bond_source_counts.items())),
        "disputed_reactions": sorted(
            meta["reaction_id"] for meta in per_reaction if meta["disputed"]),
        "disputed_rule": payload["support_domain"]["bond_rule"],
        "per_reaction": per_reaction,
        "spec_digest_recomputed": content_digest_of(dict(payload)),
    }
    manifest_path = out_dir / "features_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=1, sort_keys=True,
                   ensure_ascii=False, allow_nan=False),
        encoding="utf-8")

    print(f"WP3-A feature build complete: {len(reactions)} reactions, "
          f"{population_states} states")
    print(f"  s_g      = {s_g!r} Ha/Å (fit median rule)")
    print(f"  s_deltaE = {s_delta_e!r} Ha (fit |ΔE| p95 rule)")
    print(f"  spec digest = {spec_digest}")
    print(f"  median |g̃| check = {median_check!r} (expect ~1)")
    print(f"  disputed-5 fallback applied to: "
          f"{manifest['disputed_reactions']}")
    print(f"  wall = {manifest['wall_seconds']} s -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
