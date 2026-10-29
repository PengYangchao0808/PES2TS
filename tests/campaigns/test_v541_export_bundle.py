"""V5.4.1 WP7 CLI acceptance: ``scripts/campaigns/v541_export_bundle.py``.

S3-style deliverable wrapper around ``export_embedding_bundle``.  All
fixtures are SYNTHETIC (tmp_path): a real ``V541Model`` checkpoint, a
minimal metrics.json, a fake ``run_records.jsonl`` ledger, and synthetic
WP7 provenance / benchmark-summary JSON files — the CLI is never pointed
at real 50K data, real dev/confirm, or a GPU, and the real export runs
only after the WP7 dev evaluation (executed by the main agent).

Covered: campaign-root discipline, ledger binding (outcome + checkpoint
sha), metrics agreement, evidence-JSON completeness, the exact eight-file
whitelist + SHA printout, and every refusal exit code (0/1/2).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from pes2ts_core.pesn.offline.publishing.export import (  # noqa: E402
    EMBEDDING_BUNDLE_FILES,
    EMBEDDING_REQUIRED_PROVENANCE,
)
from scripts.campaigns import v541_export_bundle as cli  # noqa: E402

RUN_ID = "V541_T0R_E0_S17"


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture()
def v541_root(tmp_path: Path) -> Path:
    """A real-shaped campaign root: <tmp>/outputs/pesn/v541_embed_50k."""
    root = tmp_path / "outputs" / "pesn" / "v541_embed_50k"
    root.mkdir(parents=True)
    return root


def _make_run_dir(root: Path, run_id: str = RUN_ID, *,
                  embedding_id: str = "E0", seed: int = 17,
                  checkpoint: bool = True) -> Path:
    """Synthetic formal run dir: real V541Model checkpoint + metrics.json."""
    run_dir = root / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint:
        from pes2ts_core.pesn.models.v541.model import V541Model

        torch.manual_seed(seed)
        model = V541Model()
        torch.save(
            {"model": model.state_dict(), "update": 37_900},
            run_dir / "checkpoint_best.pt",
        )
    metrics = _make_metrics(run_dir, embedding_id=embedding_id, seed=seed)
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    return run_dir


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _make_metrics(run_dir: Path, *, embedding_id: str = "E0",
                  seed: int = 17) -> Dict[str, Any]:
    ckpt = run_dir / "checkpoint_best.pt"
    return {
        "schema": "pesn_key_atom_v541_train_v1",
        "embedding_id": embedding_id,
        "seed": seed,
        "run_outcome": "done",
        "checkpoint_best_sha256": _sha(ckpt) if ckpt.is_file() else "0" * 64,
        "checkpoint_best_path": str(ckpt),
    }


def _write_ledger(root: Path, run_id: str = RUN_ID, *,
                  outcome: str = "done",
                  checkpoint_sha256: Optional[str] = None) -> Path:
    """Fake run_records.jsonl: latest entry for run_id carries outcome/sha."""
    ledger = root / "run_records.jsonl"
    lines = [
        json.dumps({"run_id": run_id, "event": "attempt_start",
                    "outcome": None}, sort_keys=True),
        json.dumps({"run_id": run_id, "event": "attempt_end",
                    "outcome": outcome,
                    "checkpoint_best_sha256": checkpoint_sha256},
                   sort_keys=True),
    ]
    ledger.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return ledger


def _write_provenance(path: Path, *,
                      mutate: Optional[Any] = None) -> Path:
    doc: Dict[str, Any] = {
        "config_sha256": "a" * 64,
        "common_protocol_sha256": "b" * 64,
        "embedding_spec_sha256": "c" * 64,
        "feature_recipe_sha256": "d" * 64,
        "code_sha256": "e" * 64,
        "environment_sha256": "f" * 64,
        "git_commit": "0" * 40,
        "seeds": {17: {"outcome": "done"}, 29: {"outcome": "done"},
                  43: {"outcome": "done"}},
        "screen": {"split": "inner-val"},
        "dev_exposure": {"wp7": "one-shot frozen dev eval"},
        "confirm_read": False,
        "cost": {"gpu_hours": 1.0},
        "failures_and_censoring": {"E4": "gate_skipped (cost ratio > 2.0)"},
        "capacity_diagnostics": {"T4": "gate_skipped(no_new_embedding_gain)"},
    }
    if mutate is not None:
        mutate(doc)
    path.write_text(json.dumps(doc) + "\n", encoding="utf-8")
    return path


def _write_benchmark(path: Path, *, embedding_id: str = "E0",
                     mutate: Optional[Any] = None) -> Path:
    doc: Dict[str, Any] = {
        "winner_embedding_id": embedding_id,
        "candidates": {
            "E0": {"seeds": [17, 29, 43], "h2": "pass"},
            "E1": {"seeds": [17], "h2": "fail"},
        },
        "stages": {"T3": "gate_skipped", "T4": "gate_skipped"},
        "h3": {"passed": False},
        "confirm_read": False,
    }
    if mutate is not None:
        mutate(doc)
    path.write_text(json.dumps(doc) + "\n", encoding="utf-8")
    return path


@pytest.fixture()
def happy_env(v541_root: Path, tmp_path: Path) -> Dict[str, Path]:
    """A verified champion run + complete WP7 evidence, ready to export."""
    run_dir = _make_run_dir(v541_root)
    metrics = json.loads((run_dir / "metrics.json").read_text("utf-8"))
    _write_ledger(v541_root,
                  checkpoint_sha256=metrics["checkpoint_best_sha256"])
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    return {
        "root": v541_root,
        "run_dir": run_dir,
        "provenance": _write_provenance(evidence / "provenance.json"),
        "benchmark": _write_benchmark(evidence / "benchmark.json"),
        "evidence": evidence,
    }


def _export(env: Dict[str, Path], *extra: str,
            run_id: str = RUN_ID) -> int:
    return cli.main([
        "--run-id", run_id,
        "--output-root", str(env["root"]),
        "--provenance", str(env["provenance"]),
        "--benchmark-summary", str(env["benchmark"]),
        *extra,
    ])


# --------------------------------------------------------------------------
# happy path
# --------------------------------------------------------------------------


def test_happy_path_writes_exactly_the_whitelist(
        happy_env: Dict[str, Path],
        capsys: pytest.CaptureFixture[str]) -> None:
    code = _export(happy_env)
    out = capsys.readouterr()
    assert code == 0, out.err

    bundle = (happy_env["root"] / "exports"
              / "embedding_champion_bundle")
    present = {p.name for p in bundle.iterdir() if p.is_file()}
    assert present == set(EMBEDDING_BUNDLE_FILES)
    assert present == {
        "embedding_config.yaml", "embedding_weights.pt", "feature_schema.json",
        "provenance.json", "checksums.json", "benchmark_summary.json",
        "model_card.md", "downstream_handoff.json",
    }

    # stdout: bundle path + every sha256 entry from checksums.json
    checksums = json.loads((bundle / "checksums.json").read_text("utf-8"))
    assert "bundle: {}".format(bundle) in out.out
    for name, digest in checksums.items():
        assert "{}  {}".format(digest, name) in out.out, name

    # the exported provenance keeps every required key + the CLI's binding
    prov = json.loads((bundle / "provenance.json").read_text("utf-8"))
    for key in EMBEDDING_REQUIRED_PROVENANCE:
        assert key in prov, key
    assert prov["confirm_read"] is False
    assert prov["embedding_id"] == "E0"
    assert len(prov["checkpoint_sha256"]) == 64

    # benchmark summary round-trips
    bench = json.loads((bundle / "benchmark_summary.json").read_text("utf-8"))
    assert bench["winner_embedding_id"] == "E0"


def test_happy_path_default_model_id(happy_env: Dict[str, Path]) -> None:
    assert _export(happy_env) == 0
    bundle = happy_env["root"] / "exports" / "embedding_champion_bundle"
    w = torch.load(bundle / "embedding_weights.pt", map_location="cpu",
                   weights_only=False)
    assert w["model_id"] == "pes2ts-v541-e0-s17"
    assert w["embedding_id"] == "E0"
    assert w["has_branch"] is False


def test_explicit_model_id_and_output_dir(
        happy_env: Dict[str, Path],
        capsys: pytest.CaptureFixture[str]) -> None:
    custom = happy_env["root"] / "exports" / "chosen_bundle"
    code = _export(
        happy_env,
        "--model-id", "my-model",
        "--output-dir", str(custom),
    )
    capsys.readouterr()
    assert code == 0
    w = torch.load(custom / "embedding_weights.pt", map_location="cpu",
                   weights_only=False)
    assert w["model_id"] == "my-model"
    assert not (happy_env["root"] / "exports"
                / "embedding_champion_bundle").exists()


def test_branch_run_export(happy_env: Dict[str, Path],
                           capsys: pytest.CaptureFixture[str]) -> None:
    """An E1 champion exports the branch + gate (not the empty-branch E0)."""
    from pes2ts_core.pesn.models.v541.model import V541Model

    run_dir = happy_env["root"] / "runs" / "V541_E1_S17"
    run_dir.mkdir(parents=True)
    torch.manual_seed(17)
    model = V541Model(embedding_id="E1", embedding_init_seed=1_000_020)
    torch.save({"model": model.state_dict()},
               run_dir / "checkpoint_best.pt")
    metrics = _make_metrics(run_dir, embedding_id="E1")
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics) + "\n", encoding="utf-8")
    _write_ledger(happy_env["root"], run_id="V541_E1_S17",
                  checkpoint_sha256=metrics["checkpoint_best_sha256"])
    _write_benchmark(happy_env["benchmark"], embedding_id="E1")

    code = _export(happy_env, run_id="V541_E1_S17")
    capsys.readouterr()
    assert code == 0
    bundle = happy_env["root"] / "exports" / "embedding_champion_bundle"
    w = torch.load(bundle / "embedding_weights.pt", map_location="cpu",
                   weights_only=False)
    assert w["embedding_id"] == "E1"
    assert w["has_branch"] is True
    assert "alpha" in w["branch_state"]


# --------------------------------------------------------------------------
# ledger / metrics binding refusals (exit 1, no bundle written)
# --------------------------------------------------------------------------


def test_ledger_checkpoint_mismatch_refuses(
        happy_env: Dict[str, Path]) -> None:
    _write_ledger(happy_env["root"], checkpoint_sha256="0" * 64)
    assert _export(happy_env) == 1
    assert not (happy_env["root"] / "exports").exists()


def test_ledger_outcome_not_done_refuses(
        happy_env: Dict[str, Path],
        capsys: pytest.CaptureFixture[str]) -> None:
    metrics = json.loads(
        (happy_env["run_dir"] / "metrics.json").read_text("utf-8"))
    _write_ledger(happy_env["root"], outcome="scientific_failure",
                  checkpoint_sha256=metrics["checkpoint_best_sha256"])
    code = _export(happy_env)
    err = capsys.readouterr().err
    assert code == 1
    assert "LEDGER_OUTCOME_NOT_DONE" in err
    assert not (happy_env["root"] / "exports").exists()


def test_missing_ledger_refuses(happy_env: Dict[str, Path]) -> None:
    (happy_env["root"] / "run_records.jsonl").unlink()
    assert _export(happy_env) == 1
    assert not (happy_env["root"] / "exports").exists()


def test_ledger_last_entry_wins(happy_env: Dict[str, Path]) -> None:
    """A stale 'done' line followed by a later failure must NOT export."""
    metrics = json.loads(
        (happy_env["run_dir"] / "metrics.json").read_text("utf-8"))
    sha = metrics["checkpoint_best_sha256"]
    ledger = happy_env["root"] / "run_records.jsonl"
    ledger.write_text("\n".join([
        json.dumps({"run_id": RUN_ID, "event": "attempt_end",
                    "outcome": "done", "checkpoint_best_sha256": sha}),
        json.dumps({"run_id": RUN_ID, "event": "attempt_end",
                    "outcome": "invalid_run",
                    "checkpoint_best_sha256": sha}),
    ]) + "\n", encoding="utf-8")
    assert _export(happy_env) == 1
    assert not (happy_env["root"] / "exports").exists()


def test_metrics_embedding_drift_refuses(
        happy_env: Dict[str, Path],
        capsys: pytest.CaptureFixture[str]) -> None:
    metrics = _make_metrics(happy_env["run_dir"], embedding_id="E1")
    (happy_env["run_dir"] / "metrics.json").write_text(
        json.dumps(metrics) + "\n", encoding="utf-8")
    code = _export(happy_env)
    err = capsys.readouterr().err
    assert code == 1
    assert "METRICS_EMBEDDING_DRIFT" in err


def test_metrics_outcome_drift_refuses(
        happy_env: Dict[str, Path],
        capsys: pytest.CaptureFixture[str]) -> None:
    metrics = json.loads(
        (happy_env["run_dir"] / "metrics.json").read_text("utf-8"))
    metrics["run_outcome"] = "budget_censored"
    (happy_env["run_dir"] / "metrics.json").write_text(
        json.dumps(metrics) + "\n", encoding="utf-8")
    code = _export(happy_env)
    err = capsys.readouterr().err
    assert code == 1
    assert "METRICS_OUTCOME_NOT_DONE" in err


def test_metrics_checkpoint_drift_refuses(
        happy_env: Dict[str, Path],
        capsys: pytest.CaptureFixture[str]) -> None:
    metrics = json.loads(
        (happy_env["run_dir"] / "metrics.json").read_text("utf-8"))
    metrics["checkpoint_best_sha256"] = "0" * 64
    (happy_env["run_dir"] / "metrics.json").write_text(
        json.dumps(metrics) + "\n", encoding="utf-8")
    code = _export(happy_env)
    err = capsys.readouterr().err
    assert code == 1
    assert "METRICS_CHECKPOINT_MISMATCH" in err


# --------------------------------------------------------------------------
# run-dir / run-id / root refusals
# --------------------------------------------------------------------------


def test_missing_run_dir_refuses(happy_env: Dict[str, Path]) -> None:
    import shutil

    shutil.rmtree(happy_env["run_dir"])
    assert _export(happy_env) == 1
    assert not (happy_env["root"] / "exports").exists()


def test_missing_checkpoint_refuses(happy_env: Dict[str, Path],
                                    capsys: pytest.CaptureFixture[str]) -> None:
    (happy_env["run_dir"] / "checkpoint_best.pt").unlink()
    code = _export(happy_env)
    err = capsys.readouterr().err
    assert code == 1
    assert "CHECKPOINT_MISSING" in err


@pytest.mark.parametrize("bad_id", [
    "not_a_run_id", "V53MP_20k_C_s17", "v541_E1_S17", "V541_E1_S99",
])
def test_bad_run_id_exits_2(happy_env: Dict[str, Path],
                            capsys: pytest.CaptureFixture[str],
                            bad_id: str) -> None:
    code = _export(happy_env, run_id=bad_id)
    err = capsys.readouterr().err
    assert code == 2
    assert "error:" in err


def test_output_dir_outside_campaign_root_refuses(
        happy_env: Dict[str, Path], tmp_path: Path,
        capsys: pytest.CaptureFixture[str]) -> None:
    code = _export(
        happy_env,
        "--output-dir", str(tmp_path / "somewhere_else" / "bundle"))
    err = capsys.readouterr().err
    assert code == 1
    assert "OUTSIDE_CAMPAIGN_ROOT" in err or "CAMPAIGN_ROOT" in err
    assert not (tmp_path / "somewhere_else").exists()


def test_output_root_outside_campaign_root_refuses(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    outside = tmp_path / "not_campaign"
    outside.mkdir()
    code = cli.main([
        "--run-id", RUN_ID, "--output-root", str(outside),
        "--provenance", str(tmp_path / "p.json"),
        "--benchmark-summary", str(tmp_path / "b.json"),
    ])
    err = capsys.readouterr().err
    assert code == 1
    assert "CAMPAIGN_ROOT" in err


def test_run_id_without_seed_suffix_exits_2(
        happy_env: Dict[str, Path],
        capsys: pytest.CaptureFixture[str]) -> None:
    code = _export(happy_env, run_id="V541_E1")
    err = capsys.readouterr().err
    assert code == 2
    assert "_S<seed>" in err


# --------------------------------------------------------------------------
# evidence-JSON refusals (WP7 blocks are never fabricated)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d.pop("cost"), "PROVENANCE_INCOMPLETE"),
    (lambda d: d.pop("capacity_diagnostics"), "PROVENANCE_INCOMPLETE"),
    (lambda d: d.pop("failures_and_censoring"), "PROVENANCE_INCOMPLETE"),
    (lambda d: d.pop("seeds"), "PROVENANCE_INCOMPLETE"),
    (lambda d: d.pop("screen"), "PROVENANCE_INCOMPLETE"),
    (lambda d: d.pop("dev_exposure"), "PROVENANCE_INCOMPLETE"),
])
def test_incomplete_provenance_refuses(
        happy_env: Dict[str, Path], capsys: pytest.CaptureFixture[str],
        mutate: Any, needle: str) -> None:
    _write_provenance(happy_env["provenance"], mutate=mutate)
    code = _export(happy_env)
    err = capsys.readouterr().err
    assert code == 1
    assert needle in err
    assert not (happy_env["root"] / "exports").exists()


def test_missing_provenance_file_refuses(
        happy_env: Dict[str, Path],
        capsys: pytest.CaptureFixture[str]) -> None:
    happy_env["provenance"].unlink()
    code = _export(happy_env)
    err = capsys.readouterr().err
    assert code == 1
    assert "PROVENANCE_MISSING" in err


def test_missing_benchmark_file_refuses(
        happy_env: Dict[str, Path],
        capsys: pytest.CaptureFixture[str]) -> None:
    happy_env["benchmark"].unlink()
    code = _export(happy_env)
    err = capsys.readouterr().err
    assert code == 1
    assert "BENCHMARK_SUMMARY_MISSING" in err


def test_winner_mismatch_refused_by_exporter(
        happy_env: Dict[str, Path],
        capsys: pytest.CaptureFixture[str]) -> None:
    _write_benchmark(happy_env["benchmark"], embedding_id="E3")
    code = _export(happy_env)
    err = capsys.readouterr().err
    assert code == 1
    assert "V541_BENCHMARK_WINNER" in err
    assert not (happy_env["root"] / "exports").exists()


def test_confirm_read_true_refused_by_exporter(
        happy_env: Dict[str, Path],
        capsys: pytest.CaptureFixture[str]) -> None:
    _write_provenance(happy_env["provenance"],
                      mutate=lambda d: d.update({"confirm_read": True}))
    code = _export(happy_env)
    err = capsys.readouterr().err
    assert code == 1
    assert "V541_PROVENANCE_CONFIRM" in err
    assert not (happy_env["root"] / "exports").exists()


# --------------------------------------------------------------------------
# schema / config refusal surface (exit 2)
# --------------------------------------------------------------------------


def test_bad_config_exits_2(happy_env: Dict[str, Path],
                            tmp_path: Path,
                            capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad_config.yaml"
    bad.write_text("schema: not_v541\ncampaign_id: nope\n", encoding="utf-8")
    code = _export(happy_env, "--config", str(bad))
    err = capsys.readouterr().err
    assert code == 2
    assert "error:" in err


def test_feature_schema_override_is_used(
        happy_env: Dict[str, Path],
        capsys: pytest.CaptureFixture[str]) -> None:
    from pes2ts_core.pesn.offline.publishing.export import (
        build_embedding_feature_schema,
    )

    schema_path = happy_env["evidence"] / "feature_schema.json"
    schema = build_embedding_feature_schema("E0")
    schema_path.write_text(json.dumps(schema) + "\n", encoding="utf-8")
    code = _export(happy_env, "--feature-schema", str(schema_path))
    capsys.readouterr()
    assert code == 0
    bundle = happy_env["root"] / "exports" / "embedding_champion_bundle"
    doc = json.loads((bundle / "feature_schema.json").read_text("utf-8"))
    assert doc == schema


def test_embedding_config_override_is_used(
        happy_env: Dict[str, Path],
        capsys: pytest.CaptureFixture[str]) -> None:
    override = happy_env["evidence"] / "embedding_config.json"
    override.write_text(json.dumps(
        {"registry_row": {"family": "gin_e_chemical"},
         "residual": {"alpha0": 0.0}}) + "\n", encoding="utf-8")
    code = _export(happy_env, "--embedding-config", str(override))
    capsys.readouterr()
    assert code == 0
    bundle = happy_env["root"] / "exports" / "embedding_champion_bundle"
    import yaml as _yaml

    doc = _yaml.safe_load(
        (bundle / "embedding_config.yaml").read_text("utf-8"))
    assert doc["config"]["registry_row"] == {"family": "gin_e_chemical"}
    assert doc["embedding_id"] == "E0"


# --------------------------------------------------------------------------
# whitelist drift post-check (belt + braces beyond the exporter audit)
# --------------------------------------------------------------------------


def test_whitelist_drift_extra_file_refuses(
        happy_env: Dict[str, Path],
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    real_export = cli.export_embedding_bundle

    def _drifting_export(**kwargs: Any) -> Path:
        out = real_export(**kwargs)
        (out / "rogue_optimizer.pt").write_bytes(b"should not be here")
        return out

    monkeypatch.setattr(cli, "export_embedding_bundle", _drifting_export)
    code = _export(happy_env)
    err = capsys.readouterr().err
    assert code == 1
    assert "V541_BUNDLE_WHITELIST_DRIFT" in err
    assert "rogue_optimizer.pt" in err


def test_whitelist_drift_missing_file_refuses(
        happy_env: Dict[str, Path],
        capsys: pytest.CaptureFixture[str],
        monkeypatch: pytest.MonkeyPatch) -> None:
    real_export = cli.export_embedding_bundle

    def _forgetful_export(**kwargs: Any) -> Path:
        out = real_export(**kwargs)
        (out / "model_card.md").unlink()
        return out

    monkeypatch.setattr(cli, "export_embedding_bundle", _forgetful_export)
    code = _export(happy_env)
    err = capsys.readouterr().err
    assert code == 1
    assert "V541_BUNDLE_WHITELIST_DRIFT" in err
    assert "model_card.md" in err


# --------------------------------------------------------------------------
# module surface
# --------------------------------------------------------------------------


def test_cli_module_surface() -> None:
    assert callable(cli.main)
    assert cli.LEDGER_BASENAME == "run_records.jsonl"
    assert cli.FORMAL_RUN_SUBDIR == "runs"
    assert cli._parse_run_id("V541_E1_S29") == 29
    assert cli._embedding_id_from_run_id("V541_T0R_E0_S17") == "E0"
    assert cli._embedding_id_from_run_id("V541_TOP2_S17") is None


def test_parse_run_id_accepts_all_preregistered_seeds() -> None:
    for seed in (17, 29, 43):
        assert cli._parse_run_id("V541_E3_S{}".format(seed)) == seed


@pytest.mark.parametrize("bad", ["V541_E1", "V541_E1_s17", "V541_E1_Sx"])
def test_parse_run_id_refuses_malformed(bad: str) -> None:
    with pytest.raises(ValueError):
        cli._parse_run_id(bad)


def test_list_import_unused_type_check() -> None:
    # keep the List/Optional imports honest for linting
    _f: List[str] = []
    _o: Optional[int] = None
    assert _f == [] and _o is None
