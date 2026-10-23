"""V5.3-MP export-bundle CLI acceptance: `r3_v53mp_export_bundle.py`.

S3 deliverable wrapper around ``export_move_process_bundle``.  All fixtures
are SYNTHETIC (tmp_path): a real ``KeyAtomV53MPModel`` state-dict checkpoint,
a minimal metrics.json, and a fake run_records.jsonl ledger — the CLI is never
pointed at the real 20k run dirs, and ``_build_training_provenance`` is
monkeypatched so no multi-GB manifests are ever hashed (the real builder is
exercised by the training driver tests).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

from pes2ts_core.pesn.offline.publishing.export import (
    MOVE_PROCESS_BUNDLE_FILES,
    MOVE_PROCESS_REQUIRED_PROVENANCE,
)
from scripts.campaigns import r3_v53mp_export_bundle as cli

RUN_ID = "v53mp_20k_C_s17"
TAU = 0.87

#: all six provenance keys, fake digests (never hash the real manifests)
FAKE_PROVENANCE: Dict[str, str] = {
    "config_sha256": "a" * 64,
    "data_manifest_sha256": "b" * 64,
    "teacher_manifest_sha256": "c" * 64,
    "split_sha256": "d" * 64,
    "label_spec_sha256": "e" * 64,
    "code_commit": "f" * 40,
    # extras the real builder also returns
    "campaign_run_id": RUN_ID,
    "tier": "20k",
}


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def v53mp_root(tmp_path: Path) -> Path:
    """A real-shaped campaign root: <tmp>/outputs/pesn/v53mp (absolute)."""
    root = tmp_path / "outputs" / "pesn" / "v53mp"
    root.mkdir(parents=True)
    return root


def _make_run_dir(
    root: Path,
    run_id: str = RUN_ID,
    *,
    metrics: Optional[Dict[str, Any]] = None,
    checkpoint: bool = True,
) -> Path:
    """Synthetic run dir: real KeyAtomV53MPModel checkpoint + metrics.json."""
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint:
        torch = pytest.importorskip("torch")
        from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel

        model = KeyAtomV53MPModel()  # ~57k params, same class as the trainer
        torch.save(
            {"model": model.state_dict(), "optimizer_state": {"step": 1}},
            run_dir / "checkpoint_best.pt",
        )
    if metrics is None:
        metrics = _make_metrics(run_dir)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_dir


def _make_metrics(run_dir: Path, *, tau: Any = TAU) -> Dict[str, Any]:
    ckpt = run_dir / "checkpoint_best.pt"
    digest = (
        hashlib.sha256(ckpt.read_bytes()).hexdigest() if ckpt.is_file() else "0" * 64
    )
    return {
        "schema": "pesn_v53mp_metrics_v1",
        "arm": "C",
        "seed": 17,
        "selected_update": 900,
        "checkpoint_best_sha256": digest,
        "checkpoint_best_path": str(ckpt),
        "endpoint_head": {"tau_end": tau, "auroc": 0.87},
        "move": {"key_rmse_over_zero": {"median_rmse": 0.05}},
        "move_guard_vs_r3": {"within_rmse_guard": True},
        "ordering": {"pairwise_accuracy_full": 0.9},
        "run_outcome": "ok",
        "wall_clock_s": 123.4,
    }


def _write_ledger(
    root: Path,
    run_id: str = RUN_ID,
    *,
    status: str = "done",
    checkpoint_sha256: Optional[str] = None,
    entries: Optional[List[Dict[str, Any]]] = None,
) -> Path:
    """Fake run_records.jsonl: latest entry for run_id carries given status/sha."""
    ledger = root / "run_records.jsonl"
    lines: List[str] = []
    if entries is None:
        # an earlier 'running' line proves LAST-entry-wins semantics
        lines.append(
            json.dumps(
                {
                    "run_id": "legacyhash00000001",
                    "plan_run_id": run_id,
                    "status": "running",
                    "checkpoint_sha256": None,
                },
                sort_keys=True,
            )
        )
        lines.append(
            json.dumps(
                {
                    "run_id": "legacyhash00000001",
                    "plan_run_id": run_id,
                    "status": status,
                    "checkpoint_sha256": checkpoint_sha256,
                },
                sort_keys=True,
            )
        )
    else:
        lines = [json.dumps(e, sort_keys=True) for e in entries]
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ledger


@pytest.fixture()
def provenance_calls(monkeypatch: pytest.MonkeyPatch) -> List[Dict[str, Any]]:
    """Monkeypatch _build_training_provenance: record calls, never hash data."""
    calls: List[Dict[str, Any]] = []

    def _fake(
        config: Dict[str, Any],
        config_path: Path,
        tables_dir: Path,
        split_path: Path,
        *,
        run_id: str,
        tier: str,
    ) -> Dict[str, str]:
        calls.append(
            {
                "config_path": Path(config_path),
                "tables_dir": Path(tables_dir),
                "split_path": Path(split_path),
                "run_id": run_id,
                "tier": tier,
            }
        )
        return dict(FAKE_PROVENANCE)

    monkeypatch.setattr(cli, "_build_training_provenance", _fake)
    return calls


@pytest.fixture()
def happy_env(v53mp_root: Path, provenance_calls: List[Dict[str, Any]]) -> Path:
    """Verified run (checkpoint + metrics + done ledger) ready to export."""
    run_dir = _make_run_dir(v53mp_root)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    _write_ledger(v53mp_root, checkpoint_sha256=metrics["checkpoint_best_sha256"])
    return v53mp_root


def _export(root: Path, *extra: str, run_id: str = RUN_ID) -> int:
    return cli.main(
        [
            "--run-id", run_id,
            "--output-root", str(root),
            "--config", str(
                Path(__file__).resolve().parents[1]
                / "config" / "campaigns" / "pes2ts_v53mp_20261021.yaml"
            ),
            *extra,
        ]
    )


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------


def test_happy_path_writes_exactly_whitelist(
    happy_env: Path, provenance_calls: List[Dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    code = _export(happy_env)
    out = capsys.readouterr()
    assert code == 0, out.err
    assert "WARNING" not in out.err  # arm C: no untrained-endpoint note

    bundle = happy_env / "exports" / "{}_bundle".format(RUN_ID)
    present = {p.name for p in bundle.iterdir() if p.is_file()}
    assert present == set(MOVE_PROCESS_BUNDLE_FILES)
    assert present == {
        "weights.pt", "tau_end.json", "metrics.json", "provenance.json",
        "model_card.md", "model_bundle.json", "checksums.json",
    }

    # stdout: bundle path + every sha256 entry from checksums.json
    checksums = json.loads((bundle / "checksums.json").read_text(encoding="utf-8"))
    assert "bundle: {}".format(bundle) in out.out
    for name, digest in checksums.items():
        assert "{}  {}".format(digest, name) in out.out, name

    # metrics payload is the FULL metrics.json
    src = json.loads((happy_env / RUN_ID / "metrics.json").read_text(encoding="utf-8"))
    assert json.loads((bundle / "metrics.json").read_text(encoding="utf-8")) == src

    # tau_end comes from endpoint_head.tau_end
    tau_doc = json.loads((bundle / "tau_end.json").read_text(encoding="utf-8"))
    assert tau_doc["tau_end"] == TAU

    # provenance carries all six required keys + defaults resolved sanely
    prov = json.loads((bundle / "provenance.json").read_text(encoding="utf-8"))
    for key in MOVE_PROCESS_REQUIRED_PROVENANCE:
        assert prov[key] == FAKE_PROVENANCE[key], key
    assert len(provenance_calls) == 1
    assert provenance_calls[0]["run_id"] == RUN_ID
    assert provenance_calls[0]["tier"] == "20k"
    assert provenance_calls[0]["split_path"].name == "split_manifest_v53mp.json"


def test_happy_path_default_model_id(happy_env: Path) -> None:
    assert _export(happy_env) == 0
    bundle = happy_env / "exports" / "{}_bundle".format(RUN_ID)
    doc = json.loads((bundle / "model_bundle.json").read_text(encoding="utf-8"))
    assert doc["model_id"] == "pes2ts-v53mp-C-20k-s17"


def test_explicit_model_id_and_output_dir(
    happy_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    custom = happy_env / "exports" / "chosen_bundle"
    code = _export(
        happy_env,
        "--model-id", "my-model",
        "--output-dir", str(custom),
    )
    capsys.readouterr()
    assert code == 0
    doc = json.loads((custom / "model_bundle.json").read_text(encoding="utf-8"))
    assert doc["model_id"] == "my-model"
    assert not (happy_env / "exports" / "{}_bundle".format(RUN_ID)).exists()


def test_arm_a_export_is_allowed_but_loud(
    v53mp_root: Path, provenance_calls: List[Dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_id = "v53mp_20k_A_s17"
    run_dir = _make_run_dir(v53mp_root, run_id)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    _write_ledger(v53mp_root, run_id, checkpoint_sha256=metrics["checkpoint_best_sha256"])

    code = _export(v53mp_root, run_id=run_id)
    out = capsys.readouterr()
    assert code == 0, out.err
    assert "WARNING" in out.err
    assert "UNTRAINED" in out.err
    assert "arm C is the expected export arm" in out.err
    bundle = v53mp_root / "exports" / "{}_bundle".format(run_id)
    assert {p.name for p in bundle.iterdir() if p.is_file()} == set(
        MOVE_PROCESS_BUNDLE_FILES
    )


# --------------------------------------------------------------------------
# ledger binding refusals (exit 1, no bundle written)
# --------------------------------------------------------------------------


def test_ledger_checkpoint_mismatch_refuses(
    v53mp_root: Path, provenance_calls: List[Dict[str, Any]]
) -> None:
    _make_run_dir(v53mp_root)
    _write_ledger(v53mp_root, checkpoint_sha256="0" * 64)  # != metrics sha
    assert _export(v53mp_root) == 1
    assert not (v53mp_root / "exports").exists()


def test_ledger_status_not_done_refuses(
    v53mp_root: Path, provenance_calls: List[Dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_dir = _make_run_dir(v53mp_root)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    _write_ledger(
        v53mp_root,
        status="interrupted",
        checkpoint_sha256=metrics["checkpoint_best_sha256"],
    )
    code = _export(v53mp_root)
    err = capsys.readouterr().err
    assert code == 1
    assert "LEDGER_STATUS_NOT_DONE" in err
    assert not (v53mp_root / "exports").exists()


def test_missing_ledger_refuses(
    v53mp_root: Path, provenance_calls: List[Dict[str, Any]]
) -> None:
    _make_run_dir(v53mp_root)
    assert _export(v53mp_root) == 1  # no run_records.jsonl at all
    assert not (v53mp_root / "exports").exists()


def test_ledger_last_entry_wins(
    v53mp_root: Path, provenance_calls: List[Dict[str, Any]]
) -> None:
    """A stale 'done' line followed by a later failure must NOT export."""
    run_dir = _make_run_dir(v53mp_root)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    sha = metrics["checkpoint_best_sha256"]
    _write_ledger(
        v53mp_root,
        entries=[
            {"plan_run_id": RUN_ID, "status": "done", "checkpoint_sha256": sha},
            {"plan_run_id": RUN_ID, "status": "failed", "checkpoint_sha256": sha},
        ],
    )
    assert _export(v53mp_root) == 1
    assert not (v53mp_root / "exports").exists()


# --------------------------------------------------------------------------
# tau + campaign-root + usage refusals
# --------------------------------------------------------------------------


def _prepare_run_with_tau(root: Path, tau: Any) -> None:
    """Create a verified run whose metrics carry the given tau_end value."""
    run_dir = root / RUN_ID
    run_dir.mkdir(parents=True, exist_ok=True)
    torch = pytest.importorskip("torch")
    from pes2ts_core.pesn.models.key_atom_heads_v3 import KeyAtomV53MPModel

    torch.save(
        {"model": KeyAtomV53MPModel().state_dict()},
        run_dir / "checkpoint_best.pt",
    )
    metrics = _make_metrics(run_dir, tau=tau)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_ledger(root, checkpoint_sha256=metrics["checkpoint_best_sha256"])


@pytest.mark.parametrize("tau", [1.5, -0.01, float("nan"), "not-a-number", None])
def test_tau_outside_unit_interval_refuses(
    v53mp_root: Path, provenance_calls: List[Dict[str, Any]],
    tau: Any, capsys: pytest.CaptureFixture[str],
) -> None:
    _prepare_run_with_tau(v53mp_root, tau)
    code = _export(v53mp_root)
    err = capsys.readouterr().err
    assert code == 1
    assert "tau_end" in err
    assert not (v53mp_root / "exports").exists()


def test_output_dir_outside_campaign_root_refuses(
    happy_env: Path, provenance_calls: List[Dict[str, Any]],
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    code = _export(
        happy_env, "--output-dir", str(tmp_path / "somewhere_else" / "bundle")
    )
    err = capsys.readouterr().err
    assert code == 1
    assert "OUTSIDE_CAMPAIGN_ROOT" in err or "CAMPAIGN_ROOT" in err
    assert not (tmp_path / "somewhere_else").exists()


def test_output_root_outside_campaign_root_refuses(
    tmp_path: Path, provenance_calls: List[Dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    outside = tmp_path / "not_campaign"
    outside.mkdir()
    code = cli.main(["--run-id", RUN_ID, "--output-root", str(outside)])
    err = capsys.readouterr().err
    assert code == 1
    assert "CAMPAIGN_ROOT" in err


@pytest.mark.parametrize(
    "bad_id", ["not_a_run_id", "v53mp_9k_C_s17", "v53mp_20k_Z_s17", "v53mp_20k_C_x"]
)
def test_bad_run_id_exits_2(
    v53mp_root: Path, provenance_calls: List[Dict[str, Any]],
    bad_id: str, capsys: pytest.CaptureFixture[str],
) -> None:
    code = _export(v53mp_root, run_id=bad_id)
    err = capsys.readouterr().err
    assert code == 2
    assert "error:" in err


def test_missing_run_dir_refuses(
    v53mp_root: Path, provenance_calls: List[Dict[str, Any]]
) -> None:
    _write_ledger(v53mp_root)  # ledger says done, but no run dir at all
    assert _export(v53mp_root) == 1
    assert not (v53mp_root / "exports").exists()


# --------------------------------------------------------------------------
# whitelist post-check (export succeeded but dir drifted)
# --------------------------------------------------------------------------


def test_whitelist_drift_extra_file_refuses(
    happy_env: Path, provenance_calls: List[Dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    """A bundle with an extra file must be a hard error naming the drift."""
    real_export = cli.export_move_process_bundle

    def _drifting_export(**kwargs: Any) -> Path:
        out = real_export(**kwargs)
        (out / "rogue_optimizer.pt").write_bytes(b"should not be here")
        return out

    monkeypatch.setattr(cli, "export_move_process_bundle", _drifting_export)
    code = _export(happy_env)
    err = capsys.readouterr().err
    assert code == 1
    assert "BUNDLE_WHITELIST_DRIFT" in err
    assert "rogue_optimizer.pt" in err


def test_whitelist_drift_missing_file_refuses(
    happy_env: Path, provenance_calls: List[Dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
) -> None:
    real_export = cli.export_move_process_bundle

    def _forgetful_export(**kwargs: Any) -> Path:
        out = real_export(**kwargs)
        (out / "model_card.md").unlink()
        return out

    monkeypatch.setattr(cli, "export_move_process_bundle", _forgetful_export)
    code = _export(happy_env)
    err = capsys.readouterr().err
    assert code == 1
    assert "BUNDLE_WHITELIST_DRIFT" in err
    assert "model_card.md" in err
