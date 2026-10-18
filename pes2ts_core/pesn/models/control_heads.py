"""Unified control heads: bounded residual + action-conditioned Q (ADR-0017 §3).

One shared encoder, two new heads, retained teacher heads untouched:

* **coordinate interaction** — each driver's encoder edge feature is combined
  with the graph-pooled state and the execution context through a shared
  per-driver MLP.  A light coordinate self-attention block (one
  ``MultiheadAttention`` layer applied per graph) is an OPTIONAL drop-in on
  top of the shared summary (``use_attention=True``), reserved for the
  post-closure ablation; the v1 baseline ships with it off.  Attention
  weights are never activity probabilities, chemical causes, or A(H)
  legality.
* **bounded residual head** — ``u_j = tanh(f_j)`` per legal driver; the last
  linear layer is zero-initialized so the initial proposal is exactly the
  Classic baseline (zero residual), leaving exploration to the registered
  perturbations.
* **action-conditioned Q head** — ``Q_proxy(S, a)`` scores a CANDIDATE
  DISPLACEMENT VECTOR (never a candidate index): each (candidate, driver)
  row is conditioned on the candidate's ``u_j`` and the driver summary, rows
  are mean-pooled per candidate, and a final MLP mixes the pooled state and
  execution context.  Q_proxy is a research-only candidate-efficiency proxy —
  not Q_MOVE, not a success probability, not production control.

The wrapper mirrors :class:`pes2ts_core.pesn.models.heads.ResearchHeads`
(parameters / train_mode / state_dict / load_state_dict / forward) so
training, checkpointing, and bundle tooling treat both identically.  Encoder
freezing for the first round is centralized in :meth:`ControlNet.freeze_encoder`.
"""

from __future__ import annotations

from typing import Final, Sequence

from pes2ts_core.pesn.models import require_torch

#: Residual/Q outputs below this norm are flagged invalid by callers, never
#: silently squashed.
CONTROL_MIN_NORM: Final[float] = 1e-6


class ControlHeads:
    """Coordinate summary + bounded residual + action-conditioned Q heads."""

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden: int = 64,
        context_dim: int,
        use_attention: bool = False,
    ) -> None:
        self.torch = require_torch()
        torch = self.torch
        nn = torch.nn
        self.feature_dim = feature_dim
        self.hidden = hidden
        self.context_dim = context_dim
        self.use_attention = use_attention

        class _ControlModule(nn.Module):
            def __init__(inner) -> None:
                super().__init__()
                inner.context_proj = nn.Linear(context_dim, hidden)
                inner.driver_summary = nn.Sequential(
                    nn.Linear(feature_dim + 2 * hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                )
                inner.attention = (
                    nn.MultiheadAttention(hidden, num_heads=1, batch_first=True)
                    if use_attention
                    else None
                )
                inner.residual_head = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                    nn.Tanh(),
                )
                # Zero-init the last linear layer: the initial residual is
                # exactly zero (Classic baseline) before any training.
                nn.init.zeros_(inner.residual_head[-2].weight)
                nn.init.zeros_(inner.residual_head[-2].bias)
                inner.q_driver = nn.Sequential(
                    nn.Linear(hidden + 1, hidden),
                    nn.SiLU(),
                )
                inner.q_final = nn.Sequential(
                    nn.Linear(2 * hidden + context_dim, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                )

            def summarize(inner, encoder_output, context_batch):
                pooled = encoder_output.pooled  # (B, hidden)
                driver_graph_id = encoder_output.driver_graph_id  # (D,)
                context_per_driver = inner.context_proj(context_batch)[
                    driver_graph_id
                ]
                summary = inner.driver_summary(
                    torch.cat(
                        [
                            encoder_output.driver_edge_feat,
                            pooled[driver_graph_id],
                            context_per_driver,
                        ],
                        dim=-1,
                    )
                )
                if inner.attention is not None:
                    graph_id = _driver_graph_ids(torch, encoder_output)
                    updated = summary.clone()
                    for graph in range(int(encoder_output.driver_counts.shape[0])):
                        rows = (graph_id == graph).nonzero(as_tuple=True)[0]
                        if rows.numel() == 0:
                            continue
                        block = summary[rows].unsqueeze(0)
                        attended, _ = inner.attention(
                            block, block, block, need_weights=False
                        )
                        updated[rows] = updated[rows] + attended.squeeze(0)
                    summary = updated
                return summary

            def residual(inner, summary):
                return inner.residual_head(summary).squeeze(-1)

            def q_scores(
                inner,
                encoder_output,
                context_batch,
                summary,
                candidate_u_rows,
                candidate_slot_index,
                driver_row_index,
                n_candidates,
            ):
                pooled = encoder_output.pooled
                if int(candidate_u_rows.shape[0]) == 0 or n_candidates == 0:
                    return pooled.new_zeros((n_candidates,))
                row_summary = summary[driver_row_index]  # (R, hidden)
                row_graph = encoder_output.driver_graph_id[driver_row_index]
                row_context = inner.context_proj(context_batch)[row_graph]
                row_feat = inner.q_driver(
                    torch.cat(
                        [row_summary, candidate_u_rows.unsqueeze(-1)], dim=-1
                    )
                ) + pooled[row_graph] + row_context
                agg = row_feat.new_zeros((n_candidates, row_feat.shape[-1]))
                agg.index_add_(0, candidate_slot_index, row_feat)
                counts = torch.zeros((n_candidates,), dtype=row_feat.dtype)
                counts.index_add_(
                    0, candidate_slot_index, torch.ones_like(candidate_u_rows)
                )
                agg = agg / counts.unsqueeze(-1).clamp(min=1.0)
                slot_graph = _slot_graph_of(
                    torch, candidate_slot_index, row_graph, n_candidates
                )
                out = inner.q_final(
                    torch.cat(
                        [
                            agg,
                            pooled[slot_graph],
                            context_batch[slot_graph],
                        ],
                        dim=-1,
                    )
                )
                return out.squeeze(-1)

        self.module = _ControlModule()

    # -- wrapper API mirroring ResearchHeads --------------------------------
    def parameters(self):
        return self.module.parameters()

    def train_mode(self, flag: bool) -> None:
        self.module.train(flag)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state) -> None:
        self.module.load_state_dict(state)

    def forward(self, encoder_output, context_batch):
        """Residual path: bounded u_j per driver + bookkeeping fields."""
        summary = self.module.summarize(encoder_output, context_batch)
        u = self.module.residual(summary)
        return {
            "residual_components": u,
            "driver_summary": summary,
            "driver_graph_id": encoder_output.driver_graph_id,
            "driver_counts": encoder_output.driver_counts,
        }

    def __call__(self, encoder_output, context_batch):
        return self.forward(encoder_output, context_batch)

    def score_candidates(
        self,
        encoder_output,
        context_batch,
        candidate_u_rows,
        candidate_slot_index,
        driver_row_index,
        n_candidates,
    ):
        """Q_proxy for a flat (candidate, driver) row layout (see module doc)."""
        summary = self.module.summarize(encoder_output, context_batch)
        return self.module.q_scores(
            encoder_output,
            context_batch,
            summary,
            candidate_u_rows,
            candidate_slot_index,
            driver_row_index,
            n_candidates,
        )


def _driver_graph_ids(torch, encoder_output):
    """(D,) graph id per driver row from the per-graph counts."""
    n_graphs = int(encoder_output.driver_counts.shape[0])
    total = int(encoder_output.driver_counts.sum().item())
    graph_id = torch.zeros((total,), dtype=torch.long)
    position = 0
    for graph in range(n_graphs):
        count = int(encoder_output.driver_counts[graph].item())
        graph_id[position : position + count] = graph
        position += count
    return graph_id


def _slot_graph_of(torch, candidate_slot_index, row_graph, n_candidates):
    """Graph id of each candidate slot (a candidate lives in one graph)."""
    graph_ids = torch.zeros((n_candidates,), dtype=torch.long)
    graph_ids.index_copy_(0, candidate_slot_index, row_graph.long())
    return graph_ids


class ControlNet:
    """Encoder + control heads with single entry points.

    ``forward_residual(features_list, contexts)`` returns the bounded
    residual components; ``forward_q(...)`` scores candidate vectors.  The
    encoder is shared verbatim with the teacher model so a v1 teacher bundle
    can initialize it (``load_encoder_state``).
    """

    def __init__(self, encoder, heads: ControlHeads) -> None:
        self.encoder = encoder
        self.heads = heads
        self.torch = encoder.torch

    @classmethod
    def build(
        cls,
        *,
        hidden: int = 64,
        n_layers: int = 3,
        n_basis: int = 16,
        context_dim: int,
        use_attention: bool = False,
    ) -> "ControlNet":
        from pes2ts_core.pesn.models.encoder import SmallGraphEncoder

        encoder = SmallGraphEncoder(hidden=hidden, n_layers=n_layers, n_basis=n_basis)
        heads = ControlHeads(
            feature_dim=3 * hidden + 1,
            hidden=hidden,
            context_dim=context_dim,
            use_attention=use_attention,
        )
        return cls(encoder, heads)

    def freeze_encoder(self) -> int:
        """First-round convention: freeze the encoder, train heads only."""
        frozen = 0
        for parameter in self.encoder.parameters():
            if parameter.requires_grad:
                parameter.requires_grad_(False)
                frozen += 1
        self.encoder.train_mode(False)
        return frozen

    def head_parameters(self):
        return list(self.heads.parameters())

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.heads.parameters())

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def train_mode(self, flag: bool) -> None:
        encoder_trainable = any(
            p.requires_grad for p in self.encoder.parameters()
        )
        self.encoder.train_mode(flag and encoder_trainable)
        self.heads.train_mode(flag)

    def state_dicts(self) -> dict[str, object]:
        return {
            "encoder": self.encoder.state_dict(),
            "control_heads": self.heads.state_dict(),
        }

    def load_state_dicts(self, states: dict[str, object]) -> None:
        self.encoder.load_state_dict(states["encoder"])
        self.heads.load_state_dict(states["control_heads"])

    def load_encoder_state(self, encoder_state) -> None:
        """Initialize the encoder from a v1 teacher bundle (plan §3 reuse)."""
        self.encoder.load_state_dict(encoder_state)

    def n_parameters(self) -> int:
        return self.encoder.n_parameters() + sum(
            p.numel() for p in self.heads.parameters()
        )

    def forward_residual(self, features_list: Sequence, context_batch):
        encoder_output = self.encoder.forward(list(features_list))
        return self.heads(encoder_output, context_batch)

    def forward_q(
        self,
        features_list: Sequence,
        context_batch,
        candidate_u_rows,
        candidate_slot_index,
        driver_row_index,
        n_candidates,
    ):
        encoder_output = self.encoder.forward(list(features_list))
        return self.heads.score_candidates(
            encoder_output,
            context_batch,
            candidate_u_rows,
            candidate_slot_index,
            driver_row_index,
            n_candidates,
        )

    def __call__(self, features_list: Sequence, context_batch):
        return self.forward_residual(features_list, context_batch)


__all__ = [
    "CONTROL_MIN_NORM",
    "ControlHeads",
    "ControlNet",
]
