"""``V5Model`` = unified Encoder + pluggable memory Core + shared Heads.

The same object serves training (teacher-forced full trajectories) and
deployment-style single-step inference/rollout (``step``).  Encoder and Heads
are identical across all 14 architectures; only the Core changes (fairness
protocol, plan §3.4).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from pes2ts_v5.models.cores import (
    CORE_META,
    build_core,
    count_params,
    resolve_core_config,
)
from pes2ts_v5.models.encoders import build_encoder
from pes2ts_v5.models.heads import MultiHead, compute_losses


class V5Model(nn.Module):
    def __init__(
        self,
        arch: str,
        encoder: nn.Module,
        core: nn.Module,
        heads: MultiHead,
        *,
        q_dim: int = 3,
        core_info: dict | None = None,
    ):
        super().__init__()
        self.arch = str(arch).upper()
        self.encoder = encoder
        self.core = core
        self.heads = heads
        self.q_dim = int(q_dim)
        self.core_info = core_info or {}
        #: batch fields this model requires (the model-registry path overwrites
        #: this with the full encoder+core declaration; legacy ``build_v5_model``
        #: callers get the framework default)
        declared = getattr(type(self.core), "REQUIRED_FIELDS", None)
        self.required_fields = frozenset(declared) if declared else frozenset(
            {"frame_feats", "static_feats", "lengths"})
        self.REQUIRED_FIELDS = self.required_fields

    # ------------------------------------------------------------------ info
    def param_report(self) -> dict:
        return {
            "arch": self.arch,
            "encoder_params": count_params(self.encoder),
            "core_params": count_params(self.core),
            "head_params": count_params(self.heads),
            "total_params": count_params(self),
            "stateful": bool(CORE_META.get(self.arch, {}).get("stateful", False)),
            "simplified": bool(CORE_META.get(self.arch, {}).get("simplified", False)),
        }

    def init_state(self, batch_size: int, device=None):
        return self.core.init_state(batch_size, device=device)

    # -------------------------------------------------------------- forward
    def encode(self, batch: dict) -> torch.Tensor:
        return self.encoder(batch["frame_feats"], batch["static_feats"])

    def forward(self, batch: dict, state=None) -> dict:
        h = self.encode(batch)
        y, state_out = self.core(
            h,
            lengths=batch.get("lengths"),
            delta_s=batch.get("delta_s"),
            state=state,
        )
        out = self.heads(y)
        out["state"] = state_out
        out["hidden"] = y
        return out

    def losses(self, out: dict, batch: dict, weights: dict | None = None) -> dict:
        return compute_losses(out, batch, weights=weights)

    # ------------------------------------------------------------ inference
    @torch.no_grad()
    def step(self, batch_step: dict, state=None):
        """Single-step inference over a history window; returns last-step output."""
        self.eval()
        h = self.encode(batch_step)
        y, state_out = self.core(
            h,
            lengths=None,
            delta_s=batch_step.get("delta_s"),
            state=state,
        )
        last = {k: (v[:, -1] if torch.is_tensor(v) and v.dim() >= 2 else v)
                for k, v in self.heads(y).items()}
        last["state"] = state_out
        return last, state_out


def build_v5_model(
    arch: str,
    model_cfg: dict,
    *,
    frame_feat_dim: int = 11,
    static_feat_dim: int = 8,
    q_dim: int = 3,
) -> tuple[V5Model, dict]:
    """Build a model from a YAML model-config block.

    Config keys (all optional except ``core_hidden_dim`` / ``core_param_budget``):
    ``encoder``, ``hidden_dim``, ``core_hidden_dim`` ("auto" allowed),
    ``core_param_budget``, ``core_layers``, ``core_kwargs``, ``process_classes``,
    ``dropout``.
    """
    arch = str(arch).upper()
    cfg = dict(model_cfg or {})
    hidden_dim = int(cfg.get("hidden_dim", 64))
    layers = int(cfg.get("core_layers", 2))
    target_hidden = cfg.get("core_hidden_dim", "auto")
    core_info: dict = {}
    if isinstance(target_hidden, str) and target_hidden.lower() == "auto":
        budget = int(cfg.get("core_param_budget", 100_000))
        core_info = resolve_core_config(arch, budget, layers=layers)
        core_hidden = core_info["hidden_dim"]
    else:
        core_hidden = int(target_hidden)
        core_info = {"hidden_dim": core_hidden, "target": None, "rel_error": None}
    encoder = build_encoder(
        cfg.get("encoder", "descriptor"),
        frame_feat_dim=frame_feat_dim,
        static_feat_dim=static_feat_dim,
        hidden_dim=hidden_dim,
        dropout=float(cfg.get("dropout", 0.0)),
    )
    core = build_core(
        arch,
        hidden_dim=core_hidden,
        layers=layers,
        dropout=float(cfg.get("dropout", 0.0)),
        **(cfg.get("core_kwargs") or {}),
    )
    heads = MultiHead(
        hidden_dim=core_hidden,
        q_dim=q_dim,
        process_classes=int(cfg.get("process_classes", 3)),
        dropout=float(cfg.get("dropout", 0.0)),
    )
    model = V5Model(arch, encoder, core, heads, q_dim=q_dim, core_info=core_info)
    info = {
        **core_info,
        "arch": arch,
        "encoder_hidden_dim": hidden_dim,
        "core_hidden_dim": core_hidden,
        "encoder": cfg.get("encoder", "descriptor"),
        "encoder_params": count_params(encoder),
        "core_params": count_params(core),
        "head_params": count_params(heads),
        "total_params": count_params(model),
        "stateful": bool(CORE_META.get(arch, {}).get("stateful", False)),
        "simplified": bool(CORE_META.get(arch, {}).get("simplified", False)),
    }
    return model, info
