#!/usr/bin/env python3
"""OPTIONAL tiny CUDA smoke for the WP5 dual-TS IQL semantics — FIXTURE LEVEL.

This script is NOT value training and must never be pointed at real data
(WP6 gate: BC frozen + RL eligibility first).  It runs the WP5 phase
machine (frozen-BC warmup → joint) and a full checkpoint round-trip on a
SYNTHETIC toy batch with tiny nets, at most ``--updates`` updates total
(default 30: 20 warmup + 10 joint — the shortened smoke phase lengths are
recorded in the smoke output per plan §I.11 "冒烟运行另记实际阶段长度").

Skips gracefully (exit 0, explicit SKIP line) when CUDA is unavailable or
too busy to allocate its tiny workspace.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SMOKE_LABEL = "fixture-level smoke (research_only; NOT value training)"


def _synthetic_rows(n_episodes: int = 3):
    import torch

    from pes2ts_core.pesn.contracts.dual_ts import (
        ACTION_SCHEMA_VERSION,
        DualTSAction,
        DualTSActionKind,
        DualTSTransition,
    )
    from pes2ts_core.pesn.offline.training.dual_ts_iql import (
        DualTSTransitionRow,
    )

    n = 6
    pairs = np.array([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 3], [1, 4]])
    directed = np.concatenate([pairs, pairs[:, ::-1]], axis=0).T
    a_max = 0.1

    def state(seed: int) -> dict:
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, 3))
        x -= x.mean(axis=0)
        return {
            "atomic_numbers": np.array([6, 1, 7, 6, 8, 1]),
            "node_scalars": rng.normal(size=(n, 5)),
            "x_angstrom": x,
            "g_scaled": rng.normal(size=(n, 3)) * 0.5,
            "atom_feats_a": rng.random((n, 4)),
            "atom_feats_b": rng.random((n, 4)),
            "edge_feats_a": rng.random((2 * len(pairs), 4)),
            "edge_feats_b": rng.random((2 * len(pairs), 4)),
            "edge_index": directed,
        }

    def move(rng):
        a = rng.normal(size=(n, 3))
        a /= np.linalg.norm(a, axis=-1, keepdims=True)
        a *= 0.03
        z = a / a_max
        r = np.linalg.norm(z, axis=-1, keepdims=True)
        u = z * (np.arctanh(np.clip(r, 1e-12, 1 - 1e-7)) / np.maximum(r, 1e-12))
        act = DualTSAction(kind=DualTSActionKind.MOVE, a_max_angstrom=a_max,
                           latent_u=u, normalized_z=z, action_angstrom=a,
                           density_version="smoke:radial_tanh_v0")
        act.validate()
        return act

    ok = {"eligible": True, "reason": "smoke"}
    rows = []
    for ep in range(n_episodes):
        episode = f"SMOKE|E{ep}"
        for k in range(3):
            last = k == 2
            rows.append(DualTSTransitionRow(
                reaction_group_id="SMOKE",
                transition=DualTSTransition(
                    transition_id=f"{episode}-t{k}",
                    state_id=f"{episode}-s{k}",
                    next_state_id=None if last else f"{episode}-s{k + 1}",
                    episode_id=episode, branch_id="A" if ep % 2 == 0 else "B",
                    proposed_action=move(np.random.default_rng(ep * 10 + k)),
                    next_observation_digest=None if last else "ab" * 32,
                    next_observation_missing_reason=(
                        "teacher_frame_isolated" if last else None),
                    boundary="teacher_endpoint" if last else "none",
                    terminal=last,
                    head_eligibility={"Q": ok, "V": ok, "actor": ok}),
                state_inputs=state(ep * 10 + k),
                reward=0.1 * (k + 1)))
    assert ACTION_SCHEMA_VERSION.endswith("_v2")
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=30,
                        help="total updates (warmup + joint), <= 50")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--workdir", type=str,
                        default="/tmp/opencode/dual_ts_iql_smoke")
    args = parser.parse_args()
    if args.updates > 50:
        parser.error("--updates must be <= 50 (fixture-level smoke cap)")

    import torch

    print(f"[dual-ts-iql-smoke] {SMOKE_LABEL}")
    # graceful skip: unavailable or too-busy CUDA never fails the lane
    if not torch.cuda.is_available():
        print("SKIP: torch.cuda.is_available() is False")
        return 0
    try:
        torch.zeros((8, 8), device=args.device)
        torch.cuda.synchronize(args.device)
    except RuntimeError as exc:
        print(f"SKIP: CUDA busy/unavailable ({exc})")
        return 0

    from pes2ts_core.pesn.contracts.dual_ts import ACTION_SCHEMA_VERSION, \
        DualTSBundle
    from pes2ts_core.pesn.offline.training.dual_ts_iql import (
        EpisodeKeyedBuffer,
        checkpoint_content_digest,
        restore_dual_ts_iql_checkpoint,
    )

    workdir = Path(args.workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    warmup = max(1, (int(args.updates) - 2) * 2 // 3)
    joint = int(args.updates) - warmup - 2   # keep 2 updates post-restore

    # -- synthetic BC actor checkpoint (identity-bound to this smoke run) --
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pes2ts_core.pesn.models.dual_egnn import DualTSModelConfig
    from pes2ts_core.pesn.models.dual_ts_policy import DualTSPolicy
    from pes2ts_core.pesn.offline.training.dual_ts_iql import (
        BC_CHECKPOINT_SCHEMA,
    )

    seed = 42
    spec_digest = "0" * 63 + "f"
    torch.manual_seed(0)
    bc_policy = DualTSPolicy(
        config=DualTSModelConfig(hidden=32, n_layers=2, d_model=16,
                                 d_element=8),
        action_max=0.1, sigma_range=(0.02, 1.0))
    bc_payload = {
        "schema_version": BC_CHECKPOINT_SCHEMA,
        "trainer_version": "pesn_dual_ts_bc_v2.0-smoke",
        "identity": {"features_spec_digest": spec_digest,
                     "action_version": ACTION_SCHEMA_VERSION,
                     "seed": seed},
        "model_state": bc_policy.state_dicts(),
    }
    bc_payload["content_digest"] = checkpoint_content_digest(torch, bc_payload)
    bc_path = workdir / "smoke_bc_actor.pt"
    torch.save(bc_payload, bc_path)

    buf = EpisodeKeyedBuffer(_synthetic_rows())
    payload = torch.load(bc_path, map_location="cpu", weights_only=False)
    bundle = DualTSBundle(
        run_id="dual_ts_iql_smoke",
        bc_parent_checkpoint_digest=str(payload["content_digest"]),
        seeds={"training": seed, "bc": seed},
        dataset_version="smoke_synthetic_v0", norm_version="smoke_v0",
        split_version="smoke_v0", action_version=ACTION_SCHEMA_VERSION,
        environment_version="smoke_v0", reward_version="smoke_v0")
    settings_mod = __import__(
        "pes2ts_core.pesn.offline.training.dual_ts_iql", fromlist=["x"])
    settings = settings_mod.DualTSIQLSettings(
        warmup_updates=warmup, joint_max_updates=joint + 2,
        batch_reactions=2,
        seed=seed, hidden=32, n_layers=2, d_model=16, d_element=8,
        device=args.device, require_cuda=True, log_every=1)
    trainer = settings_mod.DualTSIQLTrainer(
        settings=settings, buffer=buf, run_dir=workdir / "run",
        bc_checkpoint_path=bc_path, bundle=bundle,
        features_spec_digest=spec_digest, s_g=0.015440, s_delta_e=0.0788881)
    print(f"[smoke] phase lengths: warmup={warmup} "
          f"joint_max={joint + 2} (shortened from 2000/20000 — smoke "
          f"lengths recorded; 2 joint updates reserved post-restore)")

    trainer.run_updates(warmup)
    assert trainer.phase == "warmup" or trainer.phase_transitions, \
        "phase machine did not advance"
    ref = trainer._actor_frozen_reference
    assert all(torch.equal(trainer.actor.module.state_dict()[k].cpu(), ref[k])
               for k in ref), "actor changed during warmup (GPU)"
    trainer.run_updates(joint)
    assert trainer.phase == settings_mod.PHASE_JOINT
    print(f"[smoke] warmup {trainer.warmup_updates_done} updates: actor "
          f"bit-frozen OK; joint {trainer.joint_updates_done} updates ran")

    ck = workdir / "smoke_checkpoint.pt"
    trainer.save_checkpoint(ck)
    restored = restore_dual_ts_iql_checkpoint(
        ck, buffer=buf, bundle=bundle, bc_checkpoint_path=bc_path,
        features_spec_digest=spec_digest, s_g=0.015440, s_delta_e=0.0788881,
        run_dir=workdir / "run_restored")
    assert restored.phase == trainer.phase
    assert (restored.warmup_updates_done, restored.joint_updates_done) == \
        (trainer.warmup_updates_done, trainer.joint_updates_done)
    restored.run_updates(2)
    print("[smoke] checkpoint round-trip on GPU OK (restore + 2 updates)")
    record = trainer.run_record()
    print(f"[smoke] done: phase={record['phase']} "
          f"warmup={record['warmup_updates_done']} "
          f"joint={record['joint_updates_done']} "
          f"eligibility={ {h: record['eligibility_counts'][h]['indexed'] for h in ('Q', 'V', 'actor')} }")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
