"""Runner CLI contract: the shipped smoke experiment trains end-to-end on CPU.

These tests are the acceptance gate for "`python -m pes2ts_v5.runner train` works":
one YAML in, checkpoints + provenance/cost ledgers + strict-JSON summary out,
exit code 0; typed failures (blocked encoder, missing real store) exit 1.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from pes2ts_v5 import runner

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "experiments"


def _shipped(name: str) -> dict:
    from pes2ts_v5.configs.schema import config_to_dict, load_experiment_config

    return config_to_dict(load_experiment_config(CONFIG_DIR / name))


def _last_json(capsys) -> dict:
    text = capsys.readouterr().out.strip()
    assert text, "runner printed nothing"
    return json.loads(text.splitlines()[-1])


def _tiny_doc(tmp_path: Path, base: dict) -> dict:
    doc = copy.deepcopy(base)
    doc["data"]["store_manifest"] = str(tmp_path / "synthetic_store")
    doc["data"]["split_policy"] = "synthetic_fixture"
    doc["model"]["core"]["hidden_dim"] = 16
    doc["training"]["batch_size"] = 2
    doc["training"]["epochs"] = 1
    doc["evaluation"]["cpu_benchmark"] = False
    doc["evaluation"]["ood"] = []
    return doc


def test_runner_train_end_to_end(tmp_path, smoke_doc, write_yaml, capsys) -> None:
    doc = _tiny_doc(tmp_path, smoke_doc)
    config = write_yaml(doc)
    out_dir = tmp_path / "run"

    code = runner.main([
        "train", "--config", str(config), "--max-epochs", "1",
        "--max-train-batches", "2", "--out-dir", str(out_dir)])

    assert code == 0
    summary = _last_json(capsys)
    assert summary["status"] == "ok"
    assert summary["mode"] == "train"
    assert summary["experiment_id"] == doc["experiment_id"]
    assert summary["config_hash"]
    assert summary["training"]["epochs_run"] == 1
    assert summary["model"]["total_params"] > 0

    ckpt_dir = out_dir / "checkpoints" / doc["experiment_id"]
    assert (ckpt_dir / "last.pt").exists()
    assert (ckpt_dir / "best.pt").exists()
    assert (out_dir / "provenance.json").exists()
    assert (out_dir / "cost_ledger.jsonl").exists()
    assert (out_dir / "config_resolved.json").exists()
    assert (out_dir / "summary.json").exists()

    cost_rows = [json.loads(line) for line in
                 (out_dir / "cost_ledger.jsonl").read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    assert len(cost_rows) == 1
    assert {"wall_time_s", "epoch", "train_loss", "val_loss", "params",
            "device", "seed", "config_hash"} <= set(cost_rows[0])

    provenance = json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))
    assert provenance["weights_sha256"]
    assert provenance["dataset_hash"]
    assert provenance["seed"] == doc["seed"]

    # the synthetic fixture was actually materialised at the configured path
    assert (Path(doc["data"]["store_manifest"]) / "manifest.json").exists()


def test_runner_smoke_subcommand(tmp_path, smoke_doc, write_yaml, capsys) -> None:
    doc = _tiny_doc(tmp_path, smoke_doc)
    code = runner.main(["smoke", "--config", str(write_yaml(doc))])
    assert code == 0
    summary = _last_json(capsys)
    assert summary["status"] == "ok"
    assert summary["mode"] == "smoke"
    assert summary["preflight"]["loss_finite"] is True
    assert summary["preflight"]["grads_finite"] is True


def test_runner_eval_on_run_dir(tmp_path, smoke_doc, write_yaml, capsys) -> None:
    doc = _tiny_doc(tmp_path, smoke_doc)
    config = write_yaml(doc)
    out_dir = tmp_path / "run"
    assert runner.main(["train", "--config", str(config), "--max-epochs", "1",
                        "--max-train-batches", "1", "--out-dir", str(out_dir)]) == 0
    _last_json(capsys)

    code = runner.main(["eval", "--run-dir", str(out_dir)])
    assert code == 0
    summary = _last_json(capsys)
    assert summary["status"] == "ok"
    assert summary["mode"] == "eval"
    assert summary["checkpoint"]
    assert "valid" in summary["evaluation"]
    assert (out_dir / "eval.json").exists()


def test_runner_blocked_encoder_exits_1(tmp_path, smoke_doc, write_yaml, capsys) -> None:
    doc = _tiny_doc(tmp_path, smoke_doc)
    doc["model"]["encoder"] = "stregnn_v2"
    code = runner.main(["train", "--config", str(write_yaml(doc)),
                        "--out-dir", str(tmp_path / "blocked")])
    assert code == 1
    summary = _last_json(capsys)
    assert summary["status"] == "error"
    assert summary["error_type"] == "ModelBuildError"
    assert summary["code"] == "ENCODER_INPUT_CONTRACT_MISMATCH"


def test_shipped_egnn_sym_config_is_typed_blocked(tmp_path, capsys) -> None:
    """v56_egnn_sym.yaml is the honest wiring fixture: exit 1 + typed code."""
    code = runner.main(["train", "--config", str(CONFIG_DIR / "v56_egnn_sym.yaml"),
                        "--out-dir", str(tmp_path / "egnn")])
    assert code == 1
    summary = _last_json(capsys)
    assert summary["status"] == "error"
    assert summary["error_type"] == "ModelBuildError"
    assert summary["code"] == "ENCODER_INPUT_CONTRACT_MISMATCH"


def test_runner_missing_real_store_exits_1(tmp_path, smoke_doc, write_yaml,
                                           capsys) -> None:
    doc = _tiny_doc(tmp_path, smoke_doc)
    doc["data"]["split_policy"] = "group_reaction_v1"
    doc["data"]["store_manifest"] = str(tmp_path / "absent_real_store")
    code = runner.main(["train", "--config", str(write_yaml(doc)),
                        "--out-dir", str(tmp_path / "missing")])
    assert code == 1
    summary = _last_json(capsys)
    assert summary["status"] == "error"
    assert summary["error_type"] == "V55ContractNotAvailable"


def test_runner_invalid_config_exits_1(tmp_path, capsys) -> None:
    code = runner.main(["train", "--config", str(tmp_path / "nope.yaml")])
    assert code == 1
    summary = _last_json(capsys)
    assert summary["status"] == "error"
    assert summary["error_type"] == "ConfigError"


def test_shipped_baseline_gru_config_is_runnable(tmp_path, write_yaml, capsys) -> None:
    """The 'new experiment = 1 YAML + same command' proof config."""
    doc = _tiny_doc(tmp_path, _shipped("v56_baseline_gru.yaml"))
    out_dir = tmp_path / "baseline"
    code = runner.main(["train", "--config", str(write_yaml(doc)),
                        "--max-epochs", "1", "--max-train-batches", "2",
                        "--out-dir", str(out_dir)])
    assert code == 0
    summary = _last_json(capsys)
    assert summary["model"]["core_id"] == "M04"
    assert (out_dir / "provenance.json").exists()


def test_runner_sweep_contract(tmp_path, write_yaml, capsys) -> None:
    """Sweep subcommand accepts the arch-matrix YAML and a subset of cores."""
    doc = _tiny_doc(tmp_path, _shipped("v56_arch_matrix.yaml"))
    doc["training"]["epochs"] = 1
    doc["evaluation"]["cpu_benchmark"] = False
    config = write_yaml(doc, "sweep.yaml")
    code = runner.main([
        "sweep", "--config", str(config), "--arch", "gru",
        "--out-dir", str(tmp_path / "sweep"), "--max-epochs", "1",
        "--max-train-batches", "1"])
    summary = _last_json(capsys)
    assert "archs" in summary
    assert summary["archs"] == ["M04"]
    assert code == 0
    assert summary["n_failed"] == 0
    assert summary["n_done"] == 1


def test_runner_is_reproducible(tmp_path, smoke_doc, write_yaml, capsys) -> None:
    """Same config + same seed => identical weights hash and val loss (P3)."""
    doc = _tiny_doc(tmp_path, smoke_doc)
    config = write_yaml(doc)
    summaries = []
    for name in ("first", "second"):
        out_dir = tmp_path / name
        code = runner.main(["train", "--config", str(config), "--max-epochs", "1",
                            "--max-train-batches", "1", "--out-dir", str(out_dir)])
        assert code == 0
        _last_json(capsys)
        summaries.append((out_dir, json.loads(
            (out_dir / "summary.json").read_text(encoding="utf-8")),
            json.loads((out_dir / "provenance.json").read_text(encoding="utf-8"))))
    (_, summary_a, prov_a), (_, summary_b, prov_b) = summaries
    assert summary_a["config_hash"] == summary_b["config_hash"]
    assert summary_a["training"]["best_val_loss"] == summary_b["training"]["best_val_loss"]
    assert prov_a["weights_sha256"] == prov_b["weights_sha256"]


def test_screening_alias_generate_synthetic_only(tmp_path, smoke_doc, write_yaml,
                                                capsys) -> None:
    """The legacy `scripts/v5/run_screening.py` alias delegates to the runner."""
    from pes2ts_v5 import screening

    doc = _tiny_doc(tmp_path, smoke_doc)
    code = screening.main(["--config", str(write_yaml(doc)),
                           "--generate-synthetic-only"])
    assert code == 0
    summary = _last_json(capsys)
    assert summary["mode"] == "generate-synthetic-only"
    assert summary["data"]["dataset_version"]
    assert summary["data"]["counts"]["n_reactions"] > 0
