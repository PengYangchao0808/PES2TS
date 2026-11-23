"""R1 kinematic geometry rollout (V5.7-B T4): fixed non-quantum geometric map.

A frozen per-reaction basis maps predicted reaction-coordinate steps to XYZ;
the geometry is then updated.  This answers **offline geometric stability only**
— it is not an xTB constrained relaxation and makes no physical-reliability
claim (plan §5.2).
"""

from __future__ import annotations

import numpy as np
import torch

from pes2ts_v5.data.trajectories import build_frame_features


def _pair_distance_rms(xyz: np.ndarray) -> float:
    if xyz.shape[0] < 2:
        return 0.0
    diff = xyz[:, None, :] - xyz[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    iu = np.triu_indices(xyz.shape[0], k=1)
    return float(np.sqrt((d[iu] ** 2).mean()))


@torch.no_grad()
def run_kinematic_rollout(
    model,
    dataset,
    split: str = "id",
    *,
    device="cpu",
    max_trajectories: int = 4,
    steps: int = 6,
    history_len: int = 8,
) -> dict:
    if getattr(dataset, "geom_for", None) is None:
        raise NotImplementedError("dataset has no kinematic geometry (geom.npz)")
    model.eval()
    idx = dataset.trajectory_indices(split)[:max_trajectories]
    per_traj = []
    for traj_idx in idx:
        seq = dataset.build_sequence(traj_idx)
        ref_q = np.asarray(seq["q"], dtype=np.float32)
        length = int(seq["length"])
        n_steps = max(1, min(int(steps), length - 1))
        basis, x0, n_atoms = dataset.geom_for(seq["reaction_row"])
        # orthonormal columns -> pseudo-inverse is the transpose
        pinv = np.linalg.pinv(basis)
        ds_step = float(np.median(np.asarray(seq["delta_s"][1:], dtype=np.float32)))
        q_cur = ref_q[0].copy()
        xyz = x0.reshape(-1) + basis @ q_cur
        static = torch.from_numpy(seq["static_feats"]).unsqueeze(0).to(device)
        frame_feats = [np.asarray(seq["frame_feats"][0], dtype=np.float32)]
        state = model.init_state(1, device=device)
        stateful = bool(getattr(model.core, "stateful", False))
        xyz_path = [xyz.copy()]
        q_path = [q_cur.copy()]
        for _ in range(n_steps):
            if stateful:
                feats = torch.from_numpy(frame_feats[-1][None, None, :]).to(device)
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
            xyz = xyz + basis @ pred.astype(np.float32)
            q_cur = (pinv @ (xyz - x0.reshape(-1))).astype(np.float32)
            q_path.append(q_cur.copy())
            xyz_path.append(xyz.copy())
            frame_feats.append(build_frame_features(
                np.stack([prev_q, q_cur]), np.asarray([0.0, ds_step], np.float32),
                np.zeros(2, np.float32), np.zeros(2, np.float32),
                input_mode="G0")[-1])
        pred_q = np.stack(q_path)
        ref = ref_q[: pred_q.shape[0]]
        q_err = np.linalg.norm(pred_q - ref, axis=-1)
        pred_xyz = np.stack(xyz_path)
        ref_xyz = x0.reshape(-1)[None, :] + np.einsum("nk,tk->tn", basis, ref)
        geo_err = np.linalg.norm(pred_xyz - ref_xyz, axis=-1) / max(n_atoms, 1) ** 0.5
        d0 = _pair_distance_rms(xyz_path[0].reshape(n_atoms, 3))
        anomalies = sum(
            1 for p in xyz_path
            if abs(_pair_distance_rms(p.reshape(n_atoms, 3)) - d0) > 0.30 * max(d0, 1e-6))
        per_traj.append({
            "trajectory_id": seq["trajectory_id"],
            "reaction_id": seq["reaction_id"],
            "n_atoms": int(n_atoms),
            "n_steps": int(n_steps),
            "mean_q_error": float(q_err.mean()),
            "final_q_error": float(q_err[-1]),
            "mean_geometry_deviation": float(geo_err.mean()),
            "anomaly_steps": int(anomalies),
        })
    q_finals = [t["final_q_error"] for t in per_traj]
    return {
        "split": split,
        "n_trajectories": len(per_traj),
        "steps": int(steps),
        "mean_final_q_error": float(np.mean(q_finals)) if q_finals else float("nan"),
        "anomaly_fraction": (
            sum(t["anomaly_steps"] for t in per_traj)
            / max(sum(t["n_steps"] for t in per_traj), 1)),
        "per_trajectory": per_traj,
        "note": "R1 = fixed non-quantum geometric map; NOT xTB relaxation; "
                "no physical-reliability claim",
    }
