"""V5.3-MP move-process bundle export acceptance (contract §6).

``export_move_process_bundle`` writes the whitelist ``pes2ts_move_process_bundle_v1``
bundle.  Refusal paths (tau range / provenance) run without torch; the
full-export tests need the ML backend (same convention as
``tests/pesn/test_bundle_export.py`` — the pesn tier runs with torch).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pes2ts_core.pesn.offline.publishing.export import (
    MOVE_PROCESS_BUNDLE_FILES,
    MOVE_PROCESS_BUNDLE_SCHEMA,
    MOVE_PROCESS_REQUIRED_PROVENANCE,
    MOVE_PROCESS_TAU_TARGET_PRECISION,
    ExportError,
    export_move_process_bundle,
)
from pes2ts_core.utils.hashing import sha256_file


def _provenance() -> dict[str, str]:
    return {
        "config_sha256": "a" * 64,
        "data_manifest_sha256": "b" * 64,
        "teacher_manifest_sha256": "c" * 64,
        "split_sha256": "d" * 64,
        "label_spec_sha256": "e" * 64,
        "code_commit": "f" * 40,
    }


def _export(run_dir: Path, out_dir: Path, **overrides):
    kwargs: dict = {
        "run_dir": run_dir,
        "output_dir": out_dir,
        "model_id": "v53mp-test-model",
        "tau_end": 0.87,
        "metrics": {"arm": "C", "move": {"rmse_zero_pct": 1.1}},
        "provenance": _provenance(),
    }
    kwargs.update(overrides)
    return export_move_process_bundle(**kwargs)


# --------------------------------------------------------------------------
# refusal paths (no torch needed)
# --------------------------------------------------------------------------


def test_export_refuses_missing_provenance_key(tmp_path: Path) -> None:
    prov = _provenance()
    del prov["split_sha256"]
    with pytest.raises(ExportError) as excinfo:
        _export(tmp_path, tmp_path / "bundle", provenance=prov)
    assert "split_sha256" in str(excinfo.value)
    assert not (tmp_path / "bundle").exists()  # refuse BEFORE any write


def test_export_refuses_empty_provenance_value(tmp_path: Path) -> None:
    prov = _provenance()
    prov["code_commit"] = ""
    with pytest.raises(ExportError):
        _export(tmp_path, tmp_path / "bundle", provenance=prov)


@pytest.mark.parametrize("tau", [-0.01, 1.01, float("nan"), float("inf")])
def test_export_refuses_tau_out_of_range(tmp_path: Path, tau: float) -> None:
    with pytest.raises(ExportError) as excinfo:
        _export(tmp_path, tmp_path / "bundle", tau_end=tau)
    assert "tau_end" in str(excinfo.value)


# --------------------------------------------------------------------------
# full export (torch backend)
# --------------------------------------------------------------------------


@pytest.fixture()
def exported_bundle(tmp_path: Path) -> Path:
    torch = pytest.importorskip("torch")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    model = torch.nn.Linear(2, 2)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer_state": {"step": 1},
            "trainer_state": {"epoch": 3},
        },
        run_dir / "checkpoint_best.pt",
    )
    return _export(run_dir, tmp_path / "bundle")


def test_bundle_contains_exactly_whitelisted_files(exported_bundle: Path) -> None:
    present = {p.name for p in exported_bundle.iterdir() if p.is_file()}
    assert present == set(MOVE_PROCESS_BUNDLE_FILES)
    assert present == {
        "weights.pt", "tau_end.json", "metrics.json", "provenance.json",
        "model_card.md", "model_bundle.json", "checksums.json",
    }


def test_weights_carry_model_state_only(exported_bundle: Path) -> None:
    torch = pytest.importorskip("torch")
    payload = torch.load(exported_bundle / "weights.pt", weights_only=True)
    assert set(payload) == {"model_state"}
    assert "optimizer_state" not in payload
    assert "trainer_state" not in payload
    assert len(payload["model_state"]) > 0  # Linear(2,2) state dicts


def test_tau_end_is_frozen_inner_val(exported_bundle: Path) -> None:
    doc = json.loads((exported_bundle / "tau_end.json").read_text(encoding="utf-8"))
    assert doc == {
        "tau_end": 0.87,
        "target_precision": MOVE_PROCESS_TAU_TARGET_PRECISION,
        "calibrated_on": "inner-val",
        "frozen": True,
    }
    assert doc["target_precision"] == 0.9


def test_provenance_carries_contract_and_heads(exported_bundle: Path) -> None:
    doc = json.loads((exported_bundle / "provenance.json").read_text(encoding="utf-8"))
    for key in MOVE_PROCESS_REQUIRED_PROVENANCE:
        assert key in doc
    assert doc["heads_version"] == "pesn_key_atom_v53mp_heads_v1"
    assert doc["backbone"] == "StarEGNNV3"
    assert doc["backbone_params"] == 55441
    assert any("MoveProcessContract" in p for p in doc["contract_docs"])


def test_bundle_json_schema_and_capabilities(exported_bundle: Path) -> None:
    doc = json.loads((exported_bundle / "model_bundle.json").read_text(encoding="utf-8"))
    assert doc["schema_version"] == MOVE_PROCESS_BUNDLE_SCHEMA
    assert doc["schema_version"] == "pes2ts_move_process_bundle_v1"
    assert doc["research_only"] is True
    assert doc["capabilities"] == {
        "move": True,
        "progress_q": "diagnostic",
        "endpoint": "diagnostic_not_deployment",
        "ordering": "diagnostic",
    }
    assert doc["tau_end"] == 0.87
    assert doc["model_id"] == "v53mp-test-model"
    assert doc["weights_sha256"] == sha256_file(exported_bundle / "weights.pt")


def test_checksums_match_recompute(exported_bundle: Path) -> None:
    checksums = json.loads(
        (exported_bundle / "checksums.json").read_text(encoding="utf-8")
    )
    assert set(checksums) == set(MOVE_PROCESS_BUNDLE_FILES) - {"checksums.json"}
    for name, digest in checksums.items():
        assert sha256_file(exported_bundle / name) == digest, name


def test_model_card_states_claim_boundaries(exported_bundle: Path) -> None:
    card = (exported_bundle / "model_card.md").read_text(encoding="utf-8")
    assert "MOVE+PROCESS" in card
    assert "NOT deployment-qualified" in card
    assert "TS_candidate" in card
    assert "endpoint_reached" in card
    assert "tau_end" in card
    assert "research_only" in card


def test_export_refuses_checkpoint_without_model_key(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    torch.save({"model_state": {"w": torch.zeros(1)}}, run_dir / "checkpoint_best.pt")
    with pytest.raises(ExportError) as excinfo:
        _export(run_dir, tmp_path / "bundle")
    assert "model" in str(excinfo.value)


def test_export_refuses_missing_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    with pytest.raises(ExportError) as excinfo:
        _export(run_dir, tmp_path / "bundle")
    assert "checkpoint" in str(excinfo.value)
