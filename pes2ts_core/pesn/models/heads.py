"""Research task heads: progress, path score, direction, activity (B4).

All heads read the encoder output only.  Signed progress uses a ``tanh``
squash; ``path_proximity`` is derived as ``|s|`` by construction (the d-vs-|s|
consistency is then structural, reported as a diagnostic, not independent
evidence).  The direction head predicts one signed component per declared
driver coordinate and is normalized per sample into a unit vector; the
activity head predicts one logit per coordinate.  Missing heads surface as
typed unsupported capabilities, never as silent zeros.
"""

from __future__ import annotations

from typing import Final

from pes2ts_core.pesn.models import require_torch

DIRECTION_MIN_NORM: Final[float] = 1e-6


class ResearchHeads:
    """Scalar + per-coordinate heads on top of the encoder output."""

    def __init__(self, *, feature_dim: int, hidden: int = 64) -> None:
        self.torch = require_torch()
        torch = self.torch
        nn = torch.nn

        class _Heads(nn.Module):
            def __init__(inner) -> None:
                super().__init__()
                inner.s_head = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                    nn.Tanh(),
                )
                inner.score_head = nn.Sequential(
                    nn.Linear(hidden, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                )
                inner.direction_head = nn.Sequential(
                    nn.Linear(feature_dim, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                )
                inner.activity_head = nn.Sequential(
                    nn.Linear(feature_dim, hidden),
                    nn.SiLU(),
                    nn.Linear(hidden, 1),
                )

            def forward(inner, encoder_output):
                pooled = encoder_output.pooled
                s_hat = inner.s_head(pooled).squeeze(-1)
                score = inner.score_head(pooled).squeeze(-1)
                u = inner.direction_head(encoder_output.driver_edge_feat).squeeze(-1)
                activity_logits = inner.activity_head(
                    encoder_output.driver_edge_feat
                ).squeeze(-1)
                return {
                    "signed_progress": s_hat,
                    "path_score": score,
                    "direction_components": u,
                    "activity_logits": activity_logits,
                    "driver_graph_id": encoder_output.driver_graph_id,
                    "driver_counts": encoder_output.driver_counts,
                }

        self.module = _Heads()

    def parameters(self):
        return self.module.parameters()

    def train_mode(self, flag: bool) -> None:
        self.module.train(flag)

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, state) -> None:
        self.module.load_state_dict(state)

    def forward(self, encoder_output):
        return self.module(encoder_output)

    def __call__(self, encoder_output):
        return self.forward(encoder_output)


class MultiTaskTeacher:
    """Encoder + research heads with a single forward entry point."""

    def __init__(self, encoder, heads) -> None:
        self.encoder = encoder
        self.heads = heads
        self.torch = encoder.torch

    def parameters(self):
        return list(self.encoder.parameters()) + list(self.heads.parameters())

    def train_mode(self, flag: bool) -> None:
        self.encoder.train_mode(flag)
        self.heads.train_mode(flag)

    def state_dicts(self) -> dict[str, object]:
        return {
            "encoder": self.encoder.state_dict(),
            "heads": self.heads.state_dict(),
        }

    def load_state_dicts(self, states: dict[str, object]) -> None:
        self.encoder.load_state_dict(states["encoder"])
        self.heads.load_state_dict(states["heads"])

    def n_parameters(self) -> int:
        return self.encoder.n_parameters() + sum(
            p.numel() for p in self.heads.parameters()
        )

    def forward(self, features_list):
        return self.heads(self.encoder.forward(features_list))

    @staticmethod
    def direction_unit_vectors(
        direction_components, driver_graph_id, driver_counts, *, n_graphs: int
    ):
        """Per-sample unit vectors from flat direction components.

        Rows with a near-zero norm are left zero and flagged invalid by the
        caller through the returned mask — never silently normalized.
        """
        torch = type(direction_components)
        unit = torch.zeros_like(direction_components)
        valid = torch.zeros_like(direction_components, dtype=torch.bool)
        start = 0
        for graph_id in range(n_graphs):
            count = int(driver_counts[graph_id].item()) if torch.is_tensor(driver_counts) else int(driver_counts[graph_id])
            end = start + count
            if count == 0:
                continue
            segment = direction_components[start:end]
            norm = segment.norm()
            if float(norm.item()) > DIRECTION_MIN_NORM:
                unit[start:end] = segment / norm
                valid[start:end] = True
            start = end
        return unit, valid


__all__ = [
    "DIRECTION_MIN_NORM",
    "MultiTaskTeacher",
    "ResearchHeads",
]
