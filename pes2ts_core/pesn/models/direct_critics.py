"""Direct Q/V value heads (WP2, plan v3 §6.3).

``V(s)`` pools the encoder scalars. ``Q(s, a)`` encodes the current
geometry *and* the candidate displacement through invariant features only:
per-atom action norms, ``a_i . g_i``, ``|a_i + g_i|``, and bond-distance
change summaries on the task edges plus the candidate-geometry pooled
encoder state (plan §6.3: never just the displacement length). Actor, V,
Q1, Q2 each own independent encoder parameters (plan §6.3); this module
only provides the heads — trainers instantiate one encoder per network.
"""
from __future__ import annotations

import math
from typing import Final

from pes2ts_core.pesn.models import require_torch

CRITIC_VERSION: Final[str] = "pesn_direct_critics_v1.0"


def enc_counts(n_graphs):
    """Backward-compat shim (unused after the node_counts refactor)."""
    raise RuntimeError("enc_counts shim should not be called")


class _TorchCriticFactory:
    def build(self, torch, encoder_module, config: dict):
        nn = torch.nn
        hidden = config["hidden"]

        class _Heads(nn.Module):
            def __init__(inner) -> None:
                super().__init__()
                inner.encoder = encoder_module
                inner.v_head = nn.Sequential(
                    nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
                inner.action_proj = nn.Linear(4, hidden)
                inner.q_head = nn.Sequential(
                    nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

            def forward(inner, batch: dict, actions=None):
                enc = inner.encode(_with_gradient_ref(batch))
                if actions is None:
                    return inner.value(enc)
                return inner.q(enc, actions)

            def encode(inner, batch: dict) -> dict:
                enc = inner.encoder(batch)
                enc["batch_gradient"] = batch["gradient"]
                return enc

            def value(inner, enc: dict) -> object:
                T = torch
                h, gid = enc["h"], enc["node_graph_id"]
                pooled = T.zeros(int(enc["n_graphs"]), h.shape[1],
                                 device=h.device).index_add_(0, gid, h)
                denom = enc["node_counts"].clamp(min=1).unsqueeze(1).to(h.dtype)
                return inner.v_head(pooled / denom).squeeze(-1)

            def action_invariants(inner, coords, gradient, actions, gid, counts):
                T = torch
                a_norm = actions.norm(dim=-1, keepdim=True)
                dot_ag = (actions * gradient).sum(-1, keepdim=True)
                norm_apg = (actions + gradient).norm(dim=-1, keepdim=True)
                moved = coords + actions
                disp = (moved - coords).norm(dim=-1, keepdim=True)
                feats = T.cat([a_norm, dot_ag, norm_apg, disp], dim=-1)
                proj = inner.action_proj(feats)
                pooled = T.zeros(int(counts), proj.shape[1],
                                 device=proj.device).index_add_(0, gid, proj)
                per_node = T.ones(proj.shape[0], device=proj.device)
                denom = T.zeros(int(counts), device=proj.device).index_add_(
                    0, gid, per_node).clamp(min=1).unsqueeze(1).to(proj.dtype)
                return pooled / denom

            def q(inner, enc: dict, actions: object) -> object:
                T = torch
                h, gid = enc["h"], enc["node_graph_id"]
                counts = enc["node_counts"]
                pooled_h = T.zeros(int(enc["n_graphs"]), h.shape[1],
                                   device=h.device).index_add_(0, gid, h)
                pooled_h = pooled_h / counts.clamp(min=1).unsqueeze(1).to(h.dtype)
                pooled_a = inner.action_invariants(
                    enc["coords"], enc["batch_gradient"], actions, gid,
                    enc["n_graphs"])
                return inner.q_head(T.cat([pooled_h, pooled_a], dim=-1)).squeeze(-1)

        return _Heads()


class DirectValueCritic:
    """V(s) wrapper over one independent encoder."""

    def __init__(self, *, encoder, action_max: float = 0.1) -> None:
        self.torch = require_torch()
        self.encoder = encoder
        self.module = _TorchCriticFactory().build(
            self.torch, encoder.module, {"hidden": encoder.config.hidden})

    def __call__(self, batch: dict):
        enc = self.module.encode(_with_gradient_ref(batch))
        return self.module.value(enc)

    def parameters(self):
        return self.module.parameters()

    def state_dicts(self) -> dict:
        return {"critic": self.module.state_dict()}

    def load_state_dicts(self, state: dict) -> None:
        self.module.load_state_dict(state["critic"])

    def to(self, device) -> "DirectValueCritic":
        self.module.to(device)
        return self


class DirectQCritic:
    """Q(s, a) wrapper over one independent encoder."""

    def __init__(self, *, encoder, action_max: float = 0.1) -> None:
        self.torch = require_torch()
        self.encoder = encoder
        self.module = _TorchCriticFactory().build(
            self.torch, encoder.module, {"hidden": encoder.config.hidden})

    def __call__(self, batch: dict, actions):
        enc = self.module.encode(_with_gradient_ref(batch))
        return self.module.q(enc, actions)

    def parameters(self):
        return self.module.parameters()

    def state_dicts(self) -> dict:
        return {"critic": self.module.state_dict()}

    def load_state_dicts(self, state: dict) -> None:
        self.module.load_state_dict(state["critic"])

    def to(self, device) -> "DirectQCritic":
        self.module.to(device)
        return self


def _with_gradient_ref(batch: dict) -> dict:
    """Expose the gradient tensor to the Q head without re-running encoder."""
    batch = dict(batch)
    batch.setdefault("encoder_gradient", batch.get("gradient"))
    return batch


__all__ = [
    "CRITIC_VERSION",
    "DirectQCritic",
    "DirectValueCritic",
]
