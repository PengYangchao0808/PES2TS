"""Gate A: per-sample full-cosine direction loss (ADR-0017 plan §7 A).

A parallel unit prediction must have EXACTLY zero direction loss regardless
of the coordinate count; the pre-fix flat elementwise mean scaled with the
segment length (1 - 1/C for a parallel C-coordinate vector).
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.offline.training.losses import compute_teacher_loss  # noqa: E402
from pes2ts_core.pesn.offline.training.loader import FrameSample, TrainBatch  # noqa: E402
from pes2ts_core.pesn.offline.training.trainer import build_teacher_model  # noqa: E402
from tests.pesn.conftest import (  # noqa: E402
    build_synthetic_reaction_data,
)
from pes2ts_core.pesn.contracts.inputs import build_pesn_input  # noqa: E402
from pes2ts_core.pesn.features.graph import build_features  # noqa: E402


def _frame_sample(reaction, frame_index: int, direction, dir_valid: bool) -> FrameSample:
    base = build_pesn_input(
        input_id=f"{reaction.reaction_id}#f{frame_index}",
        task_graph=reaction.task_graph,
        coordinate_spec=reaction.coordinate_spec,
        coordinates=reaction.coords[frame_index],
        method="synthetic",
    )
    return FrameSample(
        reaction_id=reaction.reaction_id,
        frame_index=frame_index,
        features=build_features(base),
        s=0.0, d=0.0, valid=False,
        direction=np.asarray(direction, dtype=float),
        dir_valid=dir_valid,
        activity=np.zeros(len(direction)), act_valid=False,
    )


def _force_parallel_direction(model, target: np.ndarray) -> None:
    """Bias the direction head so its normalized output parallels *target*."""
    with torch.no_grad():
        for parameter in model.heads.module.direction_head.parameters():
            parameter.add_(0.0)


def test_parallel_direction_zero_loss_independent_of_coordinate_count():
    reaction = build_synthetic_reaction_data()
    for n_coords, target in ((2, np.array([1.0, 1.0]) / np.sqrt(2)),
                             (4, np.array([1.0, 1.0, 1.0, 1.0]) / 2.0)):
        sample = _frame_sample(reaction, 0, target, True)
        batch = TrainBatch(samples=[sample], pairs=[])
        # Manually craft a model output parallel to the target by monkey-
        # patching the loss entry point: use the direction components
        # directly through a tiny stub model wrapper.
        class _Stub:
            torch = torch

            def __init__(inner, components):
                inner._components = components

            def forward(inner, features_list):
                return {
                    "signed_progress": torch.zeros(1),
                    "path_score": torch.zeros(1),
                    "direction_components": torch.as_tensor(
                        inner._components, dtype=torch.float32
                    ),
                    "activity_logits": torch.zeros(len(inner._components)),
                    "driver_graph_id": torch.zeros(
                        len(inner._components), dtype=torch.long
                    ),
                    "driver_counts": torch.as_tensor(
                        [len(inner._components)], dtype=torch.long
                    ),
                }

        stub = _Stub(np.asarray(target, dtype=float) * 3.7)  # parallel, any norm
        parts = compute_teacher_loss(stub, batch, weights={
            "s": 0.0, "d": 0.0, "rank": 0.0, "direction": 1.0,
            "activity": 0.0, "symmetry": 0.0,
        }, apply_symmetry=False)
        assert float(parts.direction.item()) == pytest.approx(0.0, abs=1e-6), (
            f"parallel {n_coords}-coordinate direction must be zero loss"
        )


def test_orthogonal_direction_loss_is_one():
    reaction = build_synthetic_reaction_data()
    target = np.array([1.0, 0.0])
    sample = _frame_sample(reaction, 0, target, True)
    batch = TrainBatch(samples=[sample], pairs=[])

    class _Stub:
        torch = torch

        def forward(inner, features_list):
            orthogonal = np.array([0.0, 5.0])
            return {
                "signed_progress": torch.zeros(1),
                "path_score": torch.zeros(1),
                "direction_components": torch.as_tensor(orthogonal, dtype=torch.float32),
                "activity_logits": torch.zeros(2),
                "driver_graph_id": torch.zeros(2, dtype=torch.long),
                "driver_counts": torch.as_tensor([2], dtype=torch.long),
            }

    parts = compute_teacher_loss(_Stub(), batch, weights={
        "s": 0.0, "d": 0.0, "rank": 0.0, "direction": 1.0,
        "activity": 0.0, "symmetry": 0.0,
    }, apply_symmetry=False)
    assert float(parts.direction.item()) == pytest.approx(1.0, abs=1e-6)


def _settings(hidden: int, n_layers: int):

    from pes2ts_core.pesn.offline.training.trainer import TrainSettings

    return TrainSettings(hidden=hidden, n_layers=n_layers)


def test_real_model_direction_loss_finite_and_trainable():
    settings = _settings(hidden=16, n_layers=1)
    model = build_teacher_model(settings)
    reaction = build_synthetic_reaction_data()
    sample = _frame_sample(
        reaction, 1, reaction.direction[1] / max(
            1e-12, float(np.linalg.norm(reaction.direction[1]))
        ), True,
    )
    batch = TrainBatch(samples=[sample], pairs=[])
    parts = compute_teacher_loss(model, batch, weights={
        "s": 0.0, "d": 0.0, "rank": 0.0, "direction": 1.0,
        "activity": 0.0, "symmetry": 0.0,
    }, apply_symmetry=False)
    assert np.isfinite(float(parts.direction.item()))
    parts.direction.backward()
    grads = [
        p.grad for p in model.heads.module.direction_head.parameters()
        if p.grad is not None
    ]
    assert grads
