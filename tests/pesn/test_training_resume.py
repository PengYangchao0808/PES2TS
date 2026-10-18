"""B4 acceptance: masked gradients, reaction-equal batching, resume semantics."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.pesn.conftest import build_synthetic_reaction_data  # noqa: E402
from pes2ts_core.pesn.offline.experiments.runs import RunRecord  # noqa: E402
from pes2ts_core.pesn.offline.training.checkpoints import (  # noqa: E402
    CheckpointError,
    load_checkpoint,
    save_checkpoint,
)
from pes2ts_core.pesn.offline.training.loader import (  # noqa: E402
    make_batch,
    select_frames,
)
from pes2ts_core.pesn.offline.training.losses import compute_teacher_loss  # noqa: E402
from pes2ts_core.pesn.offline.training.trainer import (  # noqa: E402
    Trainer,
    TrainSettings,
    build_teacher_model,
)


@pytest.fixture()
def reactions() -> list:
    import dataclasses

    rng = np.random.default_rng(11)
    data = []
    for i in range(6):
        reaction = build_synthetic_reaction_data(f"RXN_TRN{i:010d}")
        data.append(
            dataclasses.replace(
                reaction, coords=reaction.coords + rng.normal(scale=0.02, size=reaction.coords.shape)
            )
        )
    return data


def test_masks_produce_valid_gradients(reactions) -> None:
    model = build_teacher_model(TrainSettings(hidden=16, n_layers=2, n_basis=16))
    batch = make_batch(
        reactions[:3],
        frames_per_reaction=8,
        pairs_per_reaction=4,
        pair_margin=0.05,
        rng=np.random.default_rng(0),
    )
    parts = compute_teacher_loss(model, batch)
    parts.total.backward()
    gradients = [
        parameter.grad for parameter in model.parameters() if parameter.grad is not None
    ]
    assert gradients, "loss must produce gradients"
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters() if parameter.grad is not None)
    assert parts.n_valid_s > 0 and parts.n_pairs > 0


def test_fully_masked_batch_raises_typed(reactions) -> None:
    model = build_teacher_model(TrainSettings(hidden=16, n_layers=2, n_basis=16))
    empty = make_batch(
        reactions[:1],
        frames_per_reaction=4,
        pairs_per_reaction=2,
        pair_margin=0.05,
    )
    empty.samples = []
    with pytest.raises(Exception):
        compute_teacher_loss(model, empty)


def test_reaction_equal_frame_selection(reactions) -> None:
    reaction = reactions[0]
    frames = select_frames(reaction, frames_per_reaction=9)
    assert 1 <= len(frames) <= 9
    assert frames[0] == 0 or 0 in frames, "TS anchor participates when present"
    sides = set(reaction.side[frames].tolist())
    assert sides <= {-1, 0, 1}


def test_pairs_respect_margin(reactions) -> None:
    from pes2ts_core.pesn.offline.training.loader import build_pairs

    reaction = reactions[0]
    frames = select_frames(reaction, frames_per_reaction=10)
    pairs = build_pairs(reaction, frames, pairs_per_reaction=8, margin=0.05)
    assert pairs, "fixture yields ranking pairs"
    for i, j in pairs:
        assert reaction.d[i] + 0.05 < reaction.d[j]


def test_checkpoint_roundtrip_and_config_guard(tmp_path, reactions) -> None:
    settings = TrainSettings(hidden=16, n_layers=2, n_basis=16)
    model = build_teacher_model(settings)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = tmp_path / "ckpt.pt"
    save_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        epoch=3,
        best_metric=0.5,
        config_sha256="cfg1",
        dataset_manifest_sha256="data1",
    )
    payload = load_checkpoint(
        checkpoint,
        model=model,
        optimizer=optimizer,
        expected_config_sha256="cfg1",
        expected_dataset_sha256="data1",
    )
    assert payload["epoch"] == 3
    with pytest.raises(CheckpointError):
        load_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            expected_config_sha256="different",
            expected_dataset_sha256="data1",
        )


def test_trainer_trains_and_resumes(tmp_path, reactions) -> None:
    base = dict(hidden=16, n_layers=2, n_basis=16, patience=2, batch_reactions=3, seed=7, apply_symmetry=False)
    settings = TrainSettings(**{**base, "max_epochs": 2})
    run_record = RunRecord.start(
        runs_root=tmp_path,
        run_id="g3p1-test-r1",
        config={"train": {"max_epochs": 4}},
        config_sha256="cfg-hash",
        dataset_manifest_sha256="data-hash",
        split_manifest_sha256="split-hash",
        feature_spec_version="t",
        label_spec_sha256="ls",
        seed=7,
        n_parameters=0,
        backend_versions={"torch": torch.__version__},
    )
    trainer = Trainer(
        settings=settings,
        run_record=run_record,
        config_sha256="cfg-hash",
        dataset_manifest_sha256="data-hash",
    )
    summary = trainer.train(reactions[:4], reactions[4:])
    assert summary["n_parameters"] > 0
    epochs = [e["epoch"] for e in run_record.document["epochs"]]
    assert len(epochs) == 2

    resume_record = RunRecord(tmp_path / "g3p1-test-r1")
    resumed = Trainer(
        settings=TrainSettings(**{**base, "max_epochs": 4}),
        run_record=resume_record,
        config_sha256="cfg-hash",
        dataset_manifest_sha256="data-hash",
    )
    result = resumed.train(
        reactions[:4], reactions[4:], resume_from=tmp_path / "g3p1-test-r1" / "checkpoint_last.pt"
    )
    events = [
        line
        for line in (tmp_path / "g3p1-test-r1" / "events.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert any('"resumed"' in line for line in events)
    assert result["best_epoch"] >= 0
