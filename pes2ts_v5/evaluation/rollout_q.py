"""R0 low-dimensional free-run rollout (V5.7-B T4).

The model repeatedly predicts a reaction-coordinate displacement and appends it
to its own predicted path — it **never** reads future reference frames.  Compare
against the reference trajectory of the same curve/direction for report only.

G1 (cached-QC) free-run rollout is a red line (plan §2.3): feeding reference
gradients to self-generated geometries is forbidden, so this module refuses it.
"""

from __future__ import annotations

import numpy as np
import torch

from pes2ts_v5.data.trajectories import build_frame_features


@torch.no_grad()
def run_rollout_q(
    model,
    dataset,
    split: str = "id",
    *,
    device="cpu",
    max_trajectories: int = 4,
    steps: int = 8,
    history_len: int = 8,
    ds_policy: str = "median",
) -> dict:
    if str(getattr(dataset.spec, "input_mode", "G0")).upper() == "G1":
        raise PermissionError(
            "G1 free-run rollout is forbidden: reference gradients may not be "
            "attached to model-generated geometries (plan §2.3 red line)")
    model.eval()
    idx = dataset.trajectory_indices(split)[:max_trajectories]
    per_traj = []
    q_dim = int(dataset.q_dim)
    for traj_idx in idx:
        seq = dataset.build_sequence(traj_idx)
        ref_q = np.asarray(seq["q"], dtype=np.float32)
        length = int(seq["length"])
        n_steps = max(1, min(int(steps), length - 1))
        ds_step = float(np.median(np.asarray(seq["delta_s"][1:], dtype=np.float32))) \
            if length > 1 else 1.0
        static = torch.from_numpy(seq["static_feats"]).unsqueeze(0).to(device)
        q_path = [ref_q[0].copy()]
        frame_feats = [np.asarray(seq["frame_feats"][0], dtype=np.float32)]
        state = model.init_state(1, device=device)
        stateful = bool(getattr(model, "core", None) and
                        getattr(model.core, "stateful", False))
        for _ in range(n_steps):
            if stateful:
                feats = torch.from_numpy(
                    frame_feats[-1][None, None, :]).to(device)
                ds = torch.full((1, 1), ds_step, dtype=torch.float32, device=device)
                out = model({"frame_feats": feats, "static_feats": static,
                             "delta_s": ds}, state=state)
                state = out["state"]
                pred = out["move"][:, -1][0].detach().cpu().numpy()
            else:
                window = np.stack(frame_feats[-history_len:]).astype(np.float32)[None]
                ds_win = np.full((1, window.shape[1]), ds_step, dtype=np.float32)
                out = model({"frame_feats": torch.from_numpy(window).to(device),
                             "static_feats": static,
                             "delta_s": torch.from_numpy(ds_win).to(device)})
                pred = out["move"][:, -1][0].detach().cpu().numpy()
            prev_q = q_path[-1]
            next_q = (prev_q + pred).astype(np.float32)
            q_path.append(next_q)
            frame_feats.append(build_frame_features(
                np.stack([prev_q, next_q]), np.asarray([0.0, ds_step], np.float32),
                np.zeros(2, np.float32), np.zeros(2, np.float32),
                input_mode="G0")[-1])
        pred_path = np.stack(q_path)[: n_steps + 1].astype(np.float32)
        ref_path = ref_q[: pred_path.shape[0]]
        err = np.linalg.norm(pred_path[: ref_path.shape[0]] - ref_path, axis=-1)
        per_traj.append({
            "trajectory_id": seq["trajectory_id"],
            "reaction_id": seq["reaction_id"],
            "n_steps": int(n_steps),
            "step_error": [float(x) for x in err],
            "final_error": float(err[-1]) if err.size else float("nan"),
            "mean_error": float(err.mean()) if err.size else float("nan"),
            "pred_path_length": float(np.linalg.norm(np.diff(pred_path, axis=0), axis=-1).sum()),
            "ref_path_length": float(np.linalg.norm(np.diff(ref_path, axis=0), axis=-1).sum()),
        })
    finals = [t["final_error"] for t in per_traj if np.isfinite(t["final_error"])]
    means = [t["mean_error"] for t in per_traj if np.isfinite(t["mean_error"])]
    return {
        "split": split,
        "n_trajectories": len(per_traj),
        "steps": int(steps),
        "history_len": int(history_len),
        "mean_final_error": float(np.mean(finals)) if finals else float("nan"),
        "mean_step_error": float(np.mean(means)) if means else float("nan"),
        "max_final_error": float(np.max(finals)) if finals else float("nan"),
        "per_trajectory": per_traj,
        "note": "R0 = low-dimensional coordinate rollout only; not physics/Q C claim",
    }
