"""B6 acceptance: whitelist export, hash checks, backend refusal, capabilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.bundle.reader import (  # noqa: E402
    BundleError,
    load_bundle_weights,
    read_bundle_manifest,
)
from pes2ts_core.pesn.inference.predictor import Predictor  # noqa: E402
from pes2ts_core.pesn.offline.publishing.export import export_bundle  # noqa: E402
from pes2ts_core.pesn.offline.training.trainer import (  # noqa: E402
    TrainSettings,
    build_teacher_model,
)


@pytest.fixture()
def trained_bundle(tmp_path: Path) -> Path:
    settings = TrainSettings(hidden=16, n_layers=2, n_basis=16)
    model = build_teacher_model(settings)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save(
        {
            "schema_version": "pesn_checkpoint_v1",
            "epoch": 1,
            "model_state": model.state_dicts(),
            "optimizer_state": {},
        },
        run_dir / "checkpoint_best.pt",
    )
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "label_spec.json").write_text(
        json.dumps({"content_sha256": "a" * 64, "schema_version": "pesn_irc_teacher_labelspec_v1"})
    )
    (dataset_dir / "dataset_manifest.json").write_text(json.dumps({}))
    bundle_dir = tmp_path / "bundle"
    return export_bundle(
        run_dir=run_dir,
        dataset_dir=dataset_dir,
        output_dir=bundle_dir,
        model_id="pesn-irc-v0.1.0-rc.test",
        metrics={"model": {"mae_s": 0.1}},
        open_direction=True,
        open_activity=False,
        run_summary={"run_id": "r1", "status": "completed"},
        model_config={"hidden": 16, "n_layers": 2, "n_basis": 16},
    )


def test_export_contains_only_whitelisted_files(trained_bundle: Path) -> None:
    from pes2ts_core.pesn.contracts.artifacts import BUNDLE_RUNTIME_FILES

    present = {p.name for p in trained_bundle.iterdir() if p.is_file()}
    assert present <= set(BUNDLE_RUNTIME_FILES)


def test_exported_weights_carry_no_optimizer_state(trained_bundle: Path) -> None:
    payload = torch.load(trained_bundle / "weights.pt", weights_only=True)
    assert set(payload) == {"model_state"}
    assert "optimizer_state" not in payload


def test_reader_validates_hashes(trained_bundle: Path) -> None:
    manifest, root = read_bundle_manifest(trained_bundle)
    assert manifest["model_id"] == "pesn-irc-v0.1.0-rc.test"
    assert manifest["capabilities"]["teacher_direction"] == "research_only"
    assert manifest["capabilities"]["coordinate_activity"] == "disabled"

    tampered = root / "model_card.md"
    tampered.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(BundleError) as excinfo:
        read_bundle_manifest(trained_bundle)
    assert excinfo.value.code == "CHECKSUM_MISMATCH"


def test_reader_rejects_non_whitelisted_files(trained_bundle: Path) -> None:
    (trained_bundle / "teacher_labels.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BundleError) as excinfo:
        read_bundle_manifest(trained_bundle)
    assert excinfo.value.code == "BUNDLE_FILE_NOT_WHITELISTED"


def test_load_weights_requires_backend(trained_bundle: Path, monkeypatch) -> None:
    import sys

    # A None entry makes ``import torch`` raise ImportError and is restored
    # by pytest afterwards (no interpreter-shutdown inconsistency).
    monkeypatch.setitem(sys.modules, "torch", None)
    with pytest.raises(Exception) as excinfo:
        load_bundle_weights(trained_bundle)
    assert "torch" in str(excinfo.value)


def test_predictor_respects_capability_table(trained_bundle: Path) -> None:
    from tests.pesn.conftest import synthetic_coordinate_spec, synthetic_task_graph
    from pes2ts_core.pesn.contracts.inputs import build_pesn_input

    predictor = Predictor.from_bundle(trained_bundle)
    caps = predictor.capabilities()
    assert caps["progress"] == "enabled"
    assert caps["coordinate_activity"] == "disabled"

    coords = [[0.0, 0, 0], [1.3, 0.1, 0], [3.0, 0, 0], [4.1, 0, 0]]
    prediction_input = build_pesn_input(
        input_id="p",
        task_graph=synthetic_task_graph(),
        coordinate_spec=synthetic_coordinate_spec(),
        coordinates=coords,
    )
    prediction = predictor.predict_point(prediction_input)
    assert prediction.signed_progress.available
    assert prediction.coordinate_activity.available is False
    assert prediction.coordinate_activity.reason == "head_disabled"
    assert prediction.research_only is True

    batch = predictor.predict_batch([prediction_input, prediction_input])
    assert [p.input_hash for p in batch] == [
        prediction_input.input_hash,
        prediction_input.input_hash,
    ]
