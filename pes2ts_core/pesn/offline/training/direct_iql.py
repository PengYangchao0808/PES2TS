"""Direct IQL trainer (WP4, plan v3 §7.2).

Implicit Q-Learning on the direct transition stream: twin Q critics with
polyak targets, expectile value regression on the detached target-Q,
advantage-weighted policy extraction on ``log pi_z``. Effective-successor
masking follows the data: censored steps carry ``td_eligible=false`` and
never enter the TD; terminals bootstrap 0. Diagnostics report Q/V
finiteness, per-episode MC-return vs V calibration, and the effective
sample size of the AWR weights (plan §6.3/§7.2).
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Sequence

import numpy as np

from pes2ts_core.pesn.contracts.direct import DEFAULT_REWARD_SPEC, DirectRewardSpec
from pes2ts_core.pesn.models.direct_critics import DirectQCritic, DirectValueCritic
from pes2ts_core.pesn.models.direct_policy import RadialDirectPolicy, radial_inverse
from pes2ts_core.pesn.models.egnn import EGNNConfig, DirectEGNNEncoder, collate_direct_features
from pes2ts_core.pesn.offline.data.direct_dataset import DirectTransitionDataset
from pes2ts_core.pesn.offline.training.device import (
    module_summary, move_batch, resolve_device, seed_everything, set_cpu_threads,
)
from pes2ts_core.pesn.offline.training.direct_bc import sample_to_feature_inputs
from pes2ts_core.utils.jsonio import write_json

IQL_VERSION: Final[str] = "pes2ts_direct_iql_v1.0"


class DirectIQLTrainerError(RuntimeError):
    """Typed trainer refusal."""


@dataclass(frozen=True, slots=True)
class DirectIQLSettings:
    total_updates: int = 22_000
    batch_transitions: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-4
    clip: float = 1.0
    seed: int = 42
    expectile: float = 0.7
    advantage_clip: float = 10.0     # X clip on exp(A)
    awr_beta: float = 3.0            # temperature in exp(A)/beta
    target_polyak: float = 0.995
    hidden: int = 128
    n_layers: int = 4
    light: bool = False
    action_max: float = 0.1
    device: str = "cuda:0"
    require_cuda: bool = True
    cpu_intra_threads: int = 2
    cpu_inter_threads: int = 1
    eval_every: int = 1000
    log_every: int = 200

    def to_payload(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def episode_mc_returns(dataset: DirectTransitionDataset) -> dict[str, np.ndarray]:
    """Undiscounted MC return per transition (gamma = 1) within episodes."""
    out: dict[str, np.ndarray] = {}
    for rid in dataset.reaction_ids:
        samples = dataset.get_reaction(rid)
        rewards = np.asarray([s.reward for s in samples], dtype=float)
        g = np.cumsum(rewards[::-1])[::-1].copy()
        out[rid] = g
    return out


class DirectIQLTrainer:
    def __init__(
        self,
        *,
        settings: DirectIQLSettings,
        train_dataset: DirectTransitionDataset,
        run_dir: str | Path,
        config_sha256: str,
        dataset_manifest_sha256: str,
        reward_spec: DirectRewardSpec = DEFAULT_REWARD_SPEC,
    ) -> None:
        self.settings = settings
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.config_sha256 = config_sha256
        self.dataset_manifest_sha256 = dataset_manifest_sha256
        self.reward_spec = reward_spec
        self.dataset = train_dataset

        self.torch, self.device = resolve_device(
            device=settings.device, require_cuda=settings.require_cuda)
        set_cpu_threads(intra=settings.cpu_intra_threads,
                        inter=settings.cpu_inter_threads)
        seed_everything(settings.seed)
        T = self.torch
        cfg = EGNNConfig(hidden=settings.hidden, n_layers=settings.n_layers,
                         light=settings.light)
        self.policy = RadialDirectPolicy(
            encoder=DirectEGNNEncoder(cfg), action_max=settings.action_max).to(self.device)
        self.value = DirectValueCritic(encoder=DirectEGNNEncoder(cfg)).to(self.device)
        self.q1 = DirectQCritic(encoder=DirectEGNNEncoder(cfg)).to(self.device)
        self.q2 = DirectQCritic(encoder=DirectEGNNEncoder(cfg)).to(self.device)
        self.q1_target = DirectQCritic(encoder=DirectEGNNEncoder(cfg)).to(self.device)
        self.q2_target = DirectQCritic(encoder=DirectEGNNEncoder(cfg)).to(self.device)
        self.q1_target.load_state_dicts(self.q1.state_dicts())
        self.q2_target.load_state_dicts(self.q2.state_dicts())
        for p in self.q1_target.parameters():
            p.requires_grad_(False)
        for p in self.q2_target.parameters():
            p.requires_grad_(False)
        self.opt_pi = T.optim.AdamW(self.policy.parameters(), lr=settings.lr,
                                    weight_decay=settings.weight_decay)
        self.opt_v = T.optim.AdamW(self.value.parameters(), lr=settings.lr,
                                   weight_decay=settings.weight_decay)
        self.opt_q = T.optim.AdamW(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=settings.lr, weight_decay=settings.weight_decay)
        self.rng = np.random.default_rng(settings.seed)
        self._samples: dict[str, list] = {}
        self._features: dict[str, list] = {}
        for rid in train_dataset.reaction_ids:
            samples = train_dataset.get_reaction(rid)
            self._samples[rid] = samples
            self._features[rid] = [sample_to_feature_inputs(s, self.reward_spec)
                                   for s in samples]
        self._index: list[tuple[str, int, str, int]] = []
        for rid in train_dataset.reaction_ids:
            samples = self._samples[rid]
            for i, s in enumerate(samples):
                nxt = samples[i + 1] if i + 1 < len(samples) else None
                if nxt is None:
                    self._index.append((rid, i, rid, -1))      # terminal tail
                else:
                    self._index.append((rid, i, rid, i + 1))
        if not self._index:
            raise DirectIQLTrainerError("empty train dataset")
        self.mc_returns = episode_mc_returns(train_dataset)
        self.history: list[dict] = []

    # -- batching -----------------------------------------------------------

    def _sample_batch(self) -> list[tuple]:
        idx = self.rng.integers(0, len(self._index), size=self.settings.batch_transitions)
        rows = []
        for i in idx:
            rid, si, nrid, nsi = self._index[int(i)]
            s = self._samples[rid][si]
            nxt = self._samples[nrid][nsi] if nsi >= 0 else None
            rows.append((s, nxt))
        return rows

    def _featurize(self, rows: Sequence[tuple]):
        T = self.torch
        feats, u_list, rewards, boot, td_ok, next_feats = [], [], [], [], [], []
        actions = []
        for s, nxt in rows:
            feats.append(self._features[s.reaction_id][s.step_index])
            actions.append(T.as_tensor(s.action.astype(np.float32)))
            rewards.append(s.reward)
            boot.append(0.0 if s.terminated else s.bootstrap)
            td_ok.append(bool(s.td_eligible))
            if nxt is not None and s.bootstrap > 0:
                next_feats.append(self._features[nxt.reaction_id][nxt.step_index])
            else:
                next_feats.append(None)
        u_list = [radial_inverse(T, a / self.settings.action_max) for a in actions]
        return feats, u_list, rewards, boot, td_ok, next_feats

    def _batch_from(self, feats, u_list):
        batch = collate_direct_features(self.torch, feats)
        batch["expert_u"] = self.torch.cat(u_list, dim=0)
        return move_batch(batch, self.device)

    def _next_values(self, next_feats, boot):
        T = self.torch
        vals = T.zeros(len(next_feats), device=self.device)
        pending = [(i, f) for i, f in enumerate(next_feats)
                   if f is not None and boot[i] > 0]
        if pending:
            batch = move_batch(collate_direct_features(T, [f for _, f in pending]),
                               self.device)
            with T.no_grad():
                v = self.value(batch)
            for row, (i, _) in enumerate(pending):
                vals[i] = v[row]
        return vals

    # -- losses -------------------------------------------------------------

    def _expectile_loss(self, diff):
        T = self.torch
        tau = self.settings.expectile
        weight = T.where(diff >= 0, tau, 1.0 - tau)
        return (weight * diff.pow(2)).mean()

    def train(self) -> dict:
        T = self.torch
        started = time.perf_counter()
        for update in range(self.settings.total_updates):
            rows = self._sample_batch()
            feats, u_list, rewards, boot, td_ok, next_feats = self._featurize(rows)
            batch = self._batch_from(feats, u_list)
            r_t = T.as_tensor(np.asarray(rewards, dtype=np.float32), device=self.device)
            boot_t = T.as_tensor(np.asarray(boot, dtype=np.float32), device=self.device)
            td_mask = T.as_tensor(np.asarray(td_ok, dtype=np.float32), device=self.device)

            with T.no_grad():
                q_target = T.minimum(self.q1_target.module(batch, batch["expert_u"]),
                                     self.q2_target.module(batch, batch["expert_u"]))
                v_next = self._next_values(next_feats, boot)
                y = r_t + float(self.reward_spec.gamma) * boot_t * v_next

            # value update
            v_pred = self.value(batch)
            loss_v = self._expectile_loss(q_target - v_pred)
            self.opt_v.zero_grad(set_to_none=True)
            loss_v.backward()
            T.nn.utils.clip_grad_norm_(self.value.parameters(), self.settings.clip)
            self.opt_v.step()

            # q update (masked to td-eligible rows)
            q1_pred = self.q1(batch, batch["expert_u"])
            q2_pred = self.q2(batch, batch["expert_u"])
            sq = ((q1_pred - y).pow(2) + (q2_pred - y).pow(2)) * 0.5 * td_mask
            loss_q = sq.sum() / td_mask.sum().clamp(min=1.0)
            self.opt_q.zero_grad(set_to_none=True)
            loss_q.backward()
            T.nn.utils.clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()), self.settings.clip)
            self.opt_q.step()

            # policy update (advantage-weighted log pi_z)
            with T.no_grad():
                q_min = T.minimum(q1_pred, q2_pred)
                adv = (q_min - v_pred).clamp(-self.settings.advantage_clip,
                                             self.settings.advantage_clip)
                weights = T.exp(adv / self.settings.awr_beta)
            out = self.policy(batch)
            lp = self.policy.module.log_prob_z(out["mu"], out["log_sigma"],
                                               batch["expert_u"])
            per_atom = lp["per_atom"]
            per_sample = T.zeros(batch["n_graphs"], device=per_atom.device).index_add_(
                0, batch["node_graph_id"], per_atom)
            counts = T.zeros(batch["n_graphs"], device=per_atom.device).index_add_(
                0, batch["node_graph_id"], T.ones_like(per_atom))
            per_sample = per_sample / counts.clamp(min=1)
            loss_pi = -(weights * per_sample).mean()
            self.opt_pi.zero_grad(set_to_none=True)
            loss_pi.backward()
            T.nn.utils.clip_grad_norm_(self.policy.parameters(), self.settings.clip)
            self.opt_pi.step()

            with T.no_grad():
                polyak = self.settings.target_polyak
                for critic, target in ((self.q1, self.q1_target),
                                       (self.q2, self.q2_target)):
                    for p, pt in zip(critic.parameters(), target.parameters()):
                        pt.mul_(polyak).add_(p, alpha=1.0 - polyak)

            if update % self.settings.log_every == 0:
                finite = bool(T.isfinite(v_pred).all() and T.isfinite(q1_pred).all())
                ess = float((weights.sum() ** 2 / (weights ** 2).sum().clamp(min=1e-12)))
                self.history.append({
                    "update": update,
                    "loss_v": float(loss_v), "loss_q": float(loss_q),
                    "loss_pi": float(loss_pi),
                    "q_finite": finite, "awr_ess": ess,
                    "seconds": round(time.perf_counter() - started, 2),
                })
            if update and update % self.settings.eval_every == 0:
                self._save_checkpoint(update, last=True)
        self._save_checkpoint(self.settings.total_updates - 1, last=True)
        return {"history": self.history,
                "model_summary": module_summary(self.policy.module)}

    def calibration_report(self) -> dict:
        """MC-return vs V calibration summary (plan §12 Gate-IQL input)."""
        T = self.torch
        rows = []
        with T.no_grad():
            for rid in self.dataset.reaction_ids[:64]:
                samples = self.dataset.get_reaction(rid)
                feats = [sample_to_feature_inputs(s, self.reward_spec)
                         for s in samples]
                batch = move_batch(collate_direct_features(T, feats), self.device)
                v = self.value(batch).cpu().numpy()
                g = self.mc_returns[rid]
                for i in range(len(samples)):
                    rows.append((float(v[i]), float(g[i])))
        if not rows:
            return {"n": 0}
        v_arr = np.asarray([r[0] for r in rows])
        g_arr = np.asarray([r[1] for r in rows])
        return {
            "n": len(rows),
            "v_mean": float(v_arr.mean()), "mc_mean": float(g_arr.mean()),
            "mae": float(np.abs(v_arr - g_arr).mean()),
            "v_finite_all": bool(np.isfinite(v_arr).all()),
        }

    def _save_checkpoint(self, update: int, *, last: bool = True) -> None:
        T = self.torch
        payload = {
            "schema_version": "pesn_checkpoint_v1",
            "epoch": update,
            "best_metric": None,
            "config_sha256": self.config_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "model_state": {
                "policy": {k: v.detach().to("cpu") for k, v in
                           self.policy.module.state_dict().items()},
                "value": {k: v.detach().to("cpu") for k, v in
                          self.value.module.state_dict().items()},
                "q1": {k: v.detach().to("cpu") for k, v in
                       self.q1.module.state_dict().items()},
                "q2": {k: v.detach().to("cpu") for k, v in
                       self.q2.module.state_dict().items()},
            },
            "optimizer_state": {
                "pi": self.opt_pi.state_dict(), "v": self.opt_v.state_dict(),
                "q": self.opt_q.state_dict()},
            "torch_rng_state": T.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
            "extra": {"trainer_version": IQL_VERSION,
                      "settings": self.settings.to_payload()},
        }
        T.save(payload, self.run_dir / "checkpoint_last.pt")

    def write_run_record(self) -> None:
        write_json(self.run_dir / "run_record.json", {
            "schema_version": "pes2ts_training_run_v1",
            "trainer": "direct_iql",
            "trainer_version": IQL_VERSION,
            "config_sha256": self.config_sha256,
            "dataset_manifest_sha256": self.dataset_manifest_sha256,
            "settings": self.settings.to_payload(),
            "history": self.history,
            "calibration": self.calibration_report(),
        })


__all__ = [
    "IQL_VERSION",
    "DirectIQLSettings",
    "DirectIQLTrainer",
    "DirectIQLTrainerError",
    "episode_mc_returns",
]
