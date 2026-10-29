"""V5.4.1 WP2 — parity CLI end-to-end on SYNTHETIC fixtures (plan §5.3).

The CLI (``scripts/campaigns/v541_parity.py``) reads historical
checkpoints READ-ONLY and writes ONE new JSON audit report to the
caller-specified path.  These tests exercise the full CLI flow with a
synthetic checkpoint in ``tmp_path`` only — running the real
``v53mp_50k_C_s{17,29,43}`` checkpoints is the main agent's job."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from pes2ts_core.pesn.models.key_atom_heads_v3 import (  # noqa: E402
    KeyAtomV53MPModel,
)
from scripts.campaigns.v541_parity import (  # noqa: E402
    CPU_ATOL,
    DEFAULT_CHECKPOINTS,
    EXIT_FAIL,
    EXIT_PASS,
    EXIT_REFUSED,
    OUTPUT_KEYS,
    REPORT_SCHEMA,
    main,
    run_parity,
)

SEED = 17


def _write_synthetic_checkpoint(path, *, seed: int = SEED) -> dict:
    torch.manual_seed(seed)
    model = KeyAtomV53MPModel()
    payload = {
        "schema": "pesn_key_atom_loss_v53mp_v3",
        "model": model.state_dict(),
        "optimizer": {"state": {}, "param_groups": []},
        "update": 7,
        "best_val": 0.123,
        "best_update": 7,
        "config": {"schema": "synthetic"},
        "seeds": {"init": seed},
        "provenance": {"fixture": "tests/pesn/test_v541_model_parity_cli"},
    }
    torch.save(payload, str(path))
    return payload


def test_default_checkpoints_are_the_three_historical_c_runs() -> None:
    """The default inputs are exactly the three matched-E0 C checkpoints
    (string-level assertion only; no historical file is touched here)."""
    assert tuple(DEFAULT_CHECKPOINTS) == (
        "outputs/pesn/v53mp/s4_50k/v53mp_50k_C_s17/checkpoint_best.pt",
        "outputs/pesn/v53mp/s4_50k/v53mp_50k_C_s29/checkpoint_best.pt",
        "outputs/pesn/v53mp/s4_50k/v53mp_50k_C_s43/checkpoint_best.pt",
    )


def test_cli_synthetic_checkpoint_passes_end_to_end(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint_best.pt"
    payload = _write_synthetic_checkpoint(ckpt)
    out = tmp_path / "audit" / "v541_parity.json"
    rc = main(["--checkpoint", str(ckpt), "--out", str(out)])
    assert rc == EXIT_PASS

    report = json.loads(out.read_text())
    assert report["schema"] == REPORT_SCHEMA
    assert report["passed"] is True
    assert report["embedding_id"] == "E0"
    assert report["fixture"]["synthetic"] is True
    assert report["tolerance"]["require_bitwise"] is True
    assert report["tolerance"]["cpu_atol_record_only"] == CPU_ATOL
    assert report["checkpoints"] == [str(ckpt)]

    (entry,) = report["entries"]
    assert entry["passed"] is True and entry["error"] is None
    assert entry["sha256"] and len(entry["sha256"]) == 64
    assert entry["payload_schema"] == "pesn_key_atom_loss_v53mp_v3"
    assert entry["payload_update"] == 7
    audit = entry["load_audit"]
    assert audit["ok"] is True
    assert audit["missing"] == [] and audit["unexpected"] == []
    assert audit["shape_mismatch"] == [] and audit["dtype_mismatch"] == []
    assert entry["load_strict"] is True
    assert entry["load_missing_keys"] == []
    assert entry["load_unexpected_keys"] == []
    assert entry["state_keys_identical"] is True
    assert entry["weights_equal_legacy_vs_v541"] is True
    assert entry["n_parameters"] == {"legacy": 57619, "v541_e0": 57619}
    # five outputs bitwise + forward_move + q==order_score (§5.3 items 3-5)
    # (report is written with sort_keys=True → compare as sets)
    assert set(entry["outputs"]) == set(OUTPUT_KEYS) | {"all_bitwise"}
    for key in OUTPUT_KEYS:
        assert entry["outputs"][key]["bitwise"] is True, key
        assert entry["outputs"][key]["max_abs_diff"] == 0.0
    assert entry["outputs"]["all_bitwise"] is True
    assert entry["q_equals_order_score"] == {"legacy": True, "v541": True}
    for key in ("u", "z"):
        assert entry["forward_move"][key]["bitwise"] is True
    # checkpoint bytes were only READ (sha recorded; payload intact)
    assert set(payload["model"]) == set(
        torch.load(str(ckpt), map_location="cpu",
                   weights_only=True)["model"])


def test_cli_refuses_to_overwrite_existing_report(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint_best.pt"
    _write_synthetic_checkpoint(ckpt)
    out = tmp_path / "report.json"
    assert main(["--checkpoint", str(ckpt), "--out", str(out)]) == EXIT_PASS
    first_bytes = out.read_bytes()
    # same command without --force → refused, file untouched
    assert main(["--checkpoint", str(ckpt), "--out", str(out)]) == EXIT_REFUSED
    assert out.read_bytes() == first_bytes
    # with --force → rewritten
    assert main(["--checkpoint", str(ckpt), "--out", str(out),
                 "--force"]) == EXIT_PASS
    assert json.loads(out.read_text())["passed"] is True


def test_cli_out_directory_refused(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint_best.pt"
    _write_synthetic_checkpoint(ckpt)
    assert main(["--checkpoint", str(ckpt), "--out", str(tmp_path)]) == EXIT_REFUSED


def test_cli_tampered_dtype_fails_audit(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint_best.pt"
    payload = _write_synthetic_checkpoint(ckpt)
    state = {k: v.clone() for k, v in payload["model"].items()}
    key = "backbone.elem_emb.weight"
    state[key] = state[key].to(torch.float16)
    payload = dict(payload)
    payload["model"] = state
    torch.save(payload, str(ckpt))

    out = tmp_path / "report.json"
    rc = main(["--checkpoint", str(ckpt), "--out", str(out)])
    assert rc == EXIT_FAIL
    report = json.loads(out.read_text())
    assert report["passed"] is False
    (entry,) = report["entries"]
    assert entry["passed"] is False
    assert entry["load_audit"]["ok"] is False
    assert entry["load_audit"]["dtype_mismatch"][0][0] == key
    assert "strict-load audit not clean" in (entry["error"] or "")


def test_cli_tampered_shape_fails_audit(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint_best.pt"
    payload = _write_synthetic_checkpoint(ckpt)
    state = {k: v.clone() for k, v in payload["model"].items()}
    key = "backbone.z_head.weight"
    state[key] = state[key][:10, :]
    payload = dict(payload)
    payload["model"] = state
    torch.save(payload, str(ckpt))

    out = tmp_path / "report.json"
    assert main(["--checkpoint", str(ckpt), "--out", str(out)]) == EXIT_FAIL
    entry = json.loads(out.read_text())["entries"][0]
    assert entry["load_audit"]["shape_mismatch"][0][0] == key
    assert entry["load_audit"]["missing"] == []


def test_cli_missing_checkpoint_fails_gracefully(tmp_path) -> None:
    missing = tmp_path / "does_not_exist.pt"
    out = tmp_path / "report.json"
    assert main(["--checkpoint", str(missing), "--out", str(out)]) == EXIT_FAIL
    entry = json.loads(out.read_text())["entries"][0]
    assert entry["passed"] is False
    assert "not found" in entry["error"]


def test_run_parity_multi_checkpoint(tmp_path) -> None:
    c1 = tmp_path / "c17" / "checkpoint_best.pt"
    c2 = tmp_path / "c29" / "checkpoint_best.pt"
    c1.parent.mkdir()
    c2.parent.mkdir()
    _write_synthetic_checkpoint(c1, seed=17)
    _write_synthetic_checkpoint(c2, seed=29)
    report = run_parity([c1, c2], fixture_seed=17, n_graphs=2)
    assert report["passed"] is True
    assert len(report["entries"]) == 2
    assert all(e["passed"] for e in report["entries"])
    # different seeds still parity — the CLI compares each checkpoint
    # against ITS OWN loaded E0 model, not across checkpoints
    assert (report["entries"][0]["sha256"]
            != report["entries"][1]["sha256"])


def test_cli_report_json_is_serializable_and_new_artifact(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint_best.pt"
    _write_synthetic_checkpoint(ckpt)
    out = tmp_path / "audit.json"
    assert main(["--checkpoint", str(ckpt), "--out", str(out)]) == EXIT_PASS
    # round-trips through strict JSON with sort_keys — no tuples/tensors
    payload = json.dumps(json.loads(out.read_text()), sort_keys=True)
    assert REPORT_SCHEMA in payload
