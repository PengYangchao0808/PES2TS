"""WP-1 mode/propagation repair counterexamples (RL repair plan §4.2/§14).

Every test drives the REAL orchestration entry (``run_control_training`` /
``run_control_freeze``) on the offline fake-ACP fixture, never the trainer
in isolation, so a mode/identity defect anywhere in the chain — CLI override,
settings, run record, checkpoints, summaries, export — fails here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.generation.control_round import (  # noqa: E402
    ControlRoundError,
    effective_training_config,
    load_control_round_config,
    run_control_collection,
    run_control_freeze,
    run_control_training,
)
from pes2ts_core.pesn.offline.publishing.control_export import (  # noqa: E402
    ControlExportError,
    export_control_bundle,
)
from tests.test_control_round import (  # noqa: E402
    _build_two_case_inputs,
)


def _collect(tmp_path: Path):
    (manifest_path, legacy_root, selection_path, config_path) = (
        _build_two_case_inputs(tmp_path, drop_second_plan=False)
    )
    config = load_control_round_config(config_path)
    round_dir = tmp_path / "rounds" / "r1"
    from tests.test_control_step import make_transport

    transport, _ = make_transport()
    run_control_collection(
        round_dir=round_dir,
        manifest_path=manifest_path,
        legacy_root=legacy_root,
        selection_path=selection_path,
        config=config,
        transport=transport,
    )
    return config, round_dir, selection_path


def _checkpoint_mode(round_dir: Path, run_id: str) -> str:
    payload = torch.load(
        round_dir / "runs" / run_id / "checkpoint_best.pt",
        map_location="cpu",
        weights_only=False,
    )
    return payload["training_mode"]


def _q_param_slices(model_state: dict) -> dict:
    return {
        name: tensor.clone()
        for name, tensor in model_state["control_heads"].items()
        if name.startswith("q_driver") or name.startswith("q_final")
    }


def _shared_param_slices(model_state: dict) -> dict:
    return {
        name: tensor.clone()
        for name, tensor in model_state["control_heads"].items()
        if not (name.startswith("q_driver") or name.startswith("q_final"))
    }


def test_cli_mode_override_reaches_settings_checkpoint_and_summary(tmp_path):
    """Round0 defect counterexample: config defaults to cql, CLI asks bc_only."""
    config, round_dir, selection_path = _collect(tmp_path)
    assert "training_mode" not in config["control_train"]

    summary = run_control_training(
        round_dir=round_dir, config=config, selection_path=selection_path,
        training_mode="bc_only",
    )
    assert summary["training_mode"] == "bc_only"
    assert summary["q_params_changed"] is False
    assert summary["encoder_params_changed"] is False
    assert _checkpoint_mode(round_dir, summary["run_id"]) == "bc_only"
    assert "-bc-" in summary["run_id"]

    run_doc = json.loads(
        (round_dir / "runs" / summary["run_id"] / "run.json").read_text()
    )
    assert run_doc["config"]["control_train"]["training_mode"] == "bc_only"
    assert run_doc["training_mode"] == "bc_only"

    cql_summary = run_control_training(
        round_dir=round_dir, config=config, selection_path=selection_path,
        training_mode="cql",
    )
    assert cql_summary["training_mode"] == "cql"
    assert cql_summary["q_params_changed"] is True
    assert _checkpoint_mode(round_dir, cql_summary["run_id"]) == "cql"

    assert cql_summary["run_id"] != summary["run_id"]
    assert (
        cql_summary["effective_config_sha256"]
        != summary["effective_config_sha256"]
    )

    bc_state = torch.load(
        round_dir / "runs" / summary["run_id"] / "checkpoint_best.pt",
        map_location="cpu", weights_only=False,
    )["model_state"]
    cql_state = torch.load(
        round_dir / "runs" / cql_summary["run_id"] / "checkpoint_best.pt",
        map_location="cpu", weights_only=False,
    )["model_state"]
    bc_q = _q_param_slices(bc_state)
    cql_q = _q_param_slices(cql_state)
    assert any(
        not torch.equal(bc_q[name], tensor) for name, tensor in cql_q.items()
    )


def test_reverse_override_config_bc_cli_cql(tmp_path):
    config, round_dir, selection_path = _collect(tmp_path)
    config["control_train"]["training_mode"] = "bc_only"
    summary = run_control_training(
        round_dir=round_dir, config=config, selection_path=selection_path,
        training_mode="cql",
    )
    assert summary["training_mode"] == "cql"
    assert _checkpoint_mode(round_dir, summary["run_id"]) == "cql"


def test_invalid_training_mode_rejected(tmp_path):
    config, round_dir, selection_path = _collect(tmp_path)
    with pytest.raises(ControlRoundError, match="training_mode"):
        run_control_training(
            round_dir=round_dir, config=config, selection_path=selection_path,
            training_mode="dq+n-step",
        )


def test_bc_shared_layers_update_while_q_and_encoder_hold(tmp_path):
    config, round_dir, selection_path = _collect(tmp_path)
    summary = run_control_training(
        round_dir=round_dir, config=config, selection_path=selection_path,
        training_mode="bc_only",
    )
    assert summary["q_params_changed"] is False
    assert summary["encoder_frozen"] is True
    assert summary["encoder_params_changed"] is False

    if summary["n_imitation_total"] > 0:
        initial = ControlTrainerModel(config)
        trained_state = torch.load(
            round_dir / "runs" / summary["run_id"] / "checkpoint_best.pt",
            map_location="cpu", weights_only=False,
        )["model_state"]
        shared_trained = _shared_param_slices(trained_state)
        changed = any(
            not torch.equal(initial.shared[name], tensor)
            for name, tensor in shared_trained.items()
        )
        assert changed, "BC must still train the shared coordinate-summary layers"


def ControlTrainerModel(config):  # noqa: N802 - fixture helper
    from pes2ts_core.pesn.offline.control_trainer import (
        ControlTrainSettings,
        build_control_model,
    )

    import numpy as np

    effective = effective_training_config(config, "bc_only")
    np.random.seed(42)
    torch.manual_seed(42)
    model = build_control_model(ControlTrainSettings.from_config(effective))
    state = model.state_dicts()
    return type(
        "ModelSnapshot",
        (),
        {"shared": _shared_param_slices(state), "q": _q_param_slices(state)},
    )()


def test_freeze_bc_bundle_disables_q_capability(tmp_path):
    config, round_dir, selection_path = _collect(tmp_path)
    run_control_training(
        round_dir=round_dir, config=config, selection_path=selection_path,
        training_mode="bc_only",
    )
    freeze = run_control_freeze(
        round_dir=round_dir, config=config, selection_path=selection_path,
        training_mode="bc_only",
    )
    manifest = json.loads(
        (Path(freeze["bundle_path"]) / "model_bundle.json").read_text()
    )
    assert manifest["capabilities"]["q_proxy"] == "disabled"
    assert manifest["capabilities"]["residual_head"] != "disabled"

    cql_freeze = run_control_freeze(
        round_dir=round_dir, config=config, selection_path=selection_path,
        training_mode="cql",
    ) if _train_cql_for_freeze(tmp_path, round_dir, config, selection_path) else None
    if cql_freeze:
        cql_manifest = json.loads(
            (Path(cql_freeze["bundle_path"]) / "model_bundle.json").read_text()
        )
        assert cql_manifest["capabilities"]["q_proxy"] != "disabled"


def _train_cql_for_freeze(tmp_path, round_dir, config, selection_path) -> bool:
    run_control_training(
        round_dir=round_dir, config=config, selection_path=selection_path,
        training_mode="cql",
    )
    return True


def test_freeze_refuses_wrong_run_wrong_config_and_existing_bundle(tmp_path):
    config, round_dir, selection_path = _collect(tmp_path)
    summary = run_control_training(
        round_dir=round_dir, config=config, selection_path=selection_path,
        training_mode="bc_only",
    )

    tampered = json.loads(
        (round_dir / "train_summary_bc.json").read_text()
    )
    original_run_id = tampered["run_id"]
    tampered["run_id"] = original_run_id + "x"
    (round_dir / "train_summary_bc.json").write_text(json.dumps(tampered))
    with pytest.raises(ControlRoundError, match="no run directory|cannot bind"):
        run_control_freeze(
            round_dir=round_dir, config=config, selection_path=selection_path,
            training_mode="bc_only",
        )
    tampered["run_id"] = original_run_id
    tampered["effective_config_sha256"] = "0" * 64
    (round_dir / "train_summary_bc.json").write_text(json.dumps(tampered))
    with pytest.raises(ControlRoundError, match="effective config"):
        run_control_freeze(
            round_dir=round_dir, config=config, selection_path=selection_path,
            training_mode="bc_only",
        )
    tampered["effective_config_sha256"] = summary["effective_config_sha256"]
    tampered["training_mode"] = "cql"
    (round_dir / "train_summary_bc.json").write_text(json.dumps(tampered))
    with pytest.raises(ControlRoundError, match="mode"):
        run_control_freeze(
            round_dir=round_dir, config=config, selection_path=selection_path,
            training_mode="cql",
        )
    tampered["training_mode"] = "bc_only"
    tampered["transitions_digest"] = "f" * 64
    (round_dir / "train_summary_bc.json").write_text(json.dumps(tampered))
    with pytest.raises((ControlRoundError, ControlExportError)):
        run_control_freeze(
            round_dir=round_dir, config=config, selection_path=selection_path,
            training_mode="bc_only",
        )
    tampered["transitions_digest"] = summary["transitions_digest"]
    (round_dir / "train_summary_bc.json").write_text(json.dumps(tampered))

    freeze = run_control_freeze(
        round_dir=round_dir, config=config, selection_path=selection_path,
        training_mode="bc_only",
    )
    with pytest.raises(ControlExportError, match="never overwritten|already exists"):
        run_control_freeze(
            round_dir=round_dir, config=config, selection_path=selection_path,
            training_mode="bc_only",
        )
    assert (Path(freeze["bundle_path"]) / "model_bundle.json").is_file()


def test_failed_export_leaves_no_bundle_and_no_staging(tmp_path):
    config, round_dir, selection_path = _collect(tmp_path)
    summary = run_control_training(
        round_dir=round_dir, config=config, selection_path=selection_path,
        training_mode="bc_only",
    )
    run_dir = round_dir / "runs" / summary["run_id"]
    target = tmp_path / "artifacts" / "broken-bundle"
    mismatched_summary = {
        "run_id": summary["run_id"],
        "training_mode": "cql",
        "effective_config_sha256": summary["effective_config_sha256"],
        "transitions_digest": summary["transitions_digest"],
    }
    with pytest.raises(ControlExportError, match="mislabelled"):
        export_control_bundle(
            run_dir=run_dir,
            output_dir=target,
            model_id="broken-bundle",
            metrics={},
            run_summary=mismatched_summary,
            model_config={"hidden": 32, "n_layers": 2, "n_basis": 16, "context_dim": 7},
            control_spec_sha256="0" * 64,
            reward_spec_sha256="0" * 64,
        )
    assert not target.exists()
    assert not list((tmp_path / "artifacts").glob("*.staging"))
    assert not list((tmp_path / "artifacts").glob(".*.staging"))
