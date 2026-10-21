"""V5.3-MP run driver acceptance: plan-init / gate-check / authorization.

Driver under test: ``scripts/campaigns/r3_v53mp_run.py`` (Lane D).

NOTE: the S0-F gate-check's import items for the loss module and the trainer
require ``torch`` (both modules import it at module level) plus the Lane-B/C
modules themselves.  ``test_gate_check_full_pass`` is therefore structured to
SKIP (with a TODO) until that environment exists, and to PASS once the
parallel lanes + a torch-capable environment are present.  The fail-closed
behaviour (missing artifacts => non-zero exit) is exercised unconditionally.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "scripts" / "campaigns" / "r3_v53mp_run.py"
CONFIG = REPO / "config" / "campaigns" / "pes2ts_v53mp_20261021.yaml"

SIX_TABLES = (
    "irc_curve.csv",
    "irc_node.csv",
    "irc_edge.csv",
    "reaction_intent.csv",
    "curve_quality.csv",
    "sample_view.csv",
)


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(DRIVER), *args],
        capture_output=True,
        text=True,
        cwd=str(REPO),
    )


def _v53mp_root(tmp_path: Path) -> Path:
    # absolute path must contain an outputs/pesn component (campaign-root rule)
    root = tmp_path / "outputs" / "pesn" / "v53mp"
    root.mkdir(parents=True)
    return root


def _stub_data(root: Path) -> tuple[Path, Path]:
    """Create a stub S0-B data tree under ``root/data``; return (tables, split)."""
    tables = root / "data"
    tables.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for name in SIX_TABLES:
        (tables / name).write_text("curve_id\n", encoding="utf-8")
        files[name] = hashlib.sha256((tables / name).read_bytes()).hexdigest()
    (tables / "data_manifest.json").write_text(
        json.dumps({"schema": "r3_6table_v2", "files": files}, sort_keys=True),
        encoding="utf-8",
    )
    (tables / "teacher_manifest.json").write_text(
        json.dumps({"files": {}}, sort_keys=True), encoding="utf-8"
    )
    split = tables / "split_manifest_v53mp.json"
    split.write_text(
        json.dumps(
            {
                "schema": "v53mp_split_v1",
                "subsets": {
                    "pool": [
                        {"curve_id": "C1", "rank": 100},
                        {"curve_id": "C2", "rank": 1500},
                        {"curve_id": "C3", "rank": 9000},
                    ],
                    "dev": ["C2"],
                    "confirm": [],
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (tables / "truth_access_log.jsonl").write_text(
        '{"source": "stub", "purpose": "test"}\n', encoding="utf-8"
    )
    return tables, split


# --------------------------------------------------------------------------
# plan-init
# --------------------------------------------------------------------------


def test_plan_init_writes_expected_run_ids_s1_smoke(tmp_path: Path) -> None:
    from scripts.agent_ops.ledger.run_ledger import (
        is_frozen_sha,
        read_run_plan,
        verify_plan_pin,
    )

    root = _v53mp_root(tmp_path)
    tables, split = _stub_data(root)
    proc = _run([
        "--output-root", str(root), "plan-init",
        "--tier", "2k", "--tables-dir", str(tables), "--split", str(split),
        "--config", str(CONFIG),
    ])
    assert proc.returncode == 0, proc.stderr
    plan_path = root / "run_plan_v53mp_2k.jsonl"
    assert plan_path.is_file()

    plan = read_run_plan(plan_path)
    assert plan["run_ids"] == {
        "v53mp_2k_A_s17", "v53mp_2k_B_s17", "v53mp_2k_C_s17",
    }
    assert plan["mode"] == "full"
    for line in plan["by_run_id"].values():
        assert line["campaign_id"] == "PES2TS_V53MP_MOVE_PROCESS"
        assert line["phase"] == "S1"
        assert line["tier"] == "2k"
        assert line["fold"] == 0
        assert line["smoke_updates"] == 100  # config training_contract value
        for field in (
            "config_sha256", "data_sha256", "teacher_sha256",
            "features_sha256", "split_sha256",
        ):
            assert is_frozen_sha(line[field]), (line["run_id"], field)

    pin_path = Path(str(plan_path) + ".pin.json")
    assert pin_path.is_file()
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    assert pin["campaign_mode"] == "full"
    verify_plan_pin(plan_path, pin)  # must NOT raise


def test_plan_init_formal_tier_has_nine_runs(tmp_path: Path) -> None:
    from scripts.agent_ops.ledger.run_ledger import read_run_plan

    root = _v53mp_root(tmp_path)
    tables, split = _stub_data(root)
    proc = _run([
        "--output-root", str(root), "plan-init",
        "--tier", "5k", "--tables-dir", str(tables), "--split", str(split),
        "--config", str(CONFIG),
    ])
    assert proc.returncode == 0, proc.stderr
    plan = read_run_plan(root / "run_plan_v53mp_5k.jsonl")
    expected = {
        "v53mp_5k_{}_s{}".format(arm, seed)
        for arm in ("A", "B", "C") for seed in (17, 29, 43)
    }
    assert plan["run_ids"] == expected
    for line in plan["by_run_id"].values():
        assert line["phase"] == "S2"
        assert line["n_updates"] == 3400  # freeze 400 + joint 3750 (5k density scaling)


def test_plan_init_refuses_missing_data_artifacts(tmp_path: Path) -> None:
    root = _v53mp_root(tmp_path)
    empty_tables = root / "no_such_tables"
    proc = _run([
        "--output-root", str(root), "plan-init",
        "--tier", "2k", "--tables-dir", str(empty_tables),
        "--config", str(CONFIG),
    ])
    assert proc.returncode != 0
    assert "PLAN_ARTIFACT_MISSING" in proc.stderr
    assert not (root / "run_plan_v53mp_2k.jsonl").exists()


def test_plan_init_refuses_rebuild_without_force(tmp_path: Path) -> None:
    root = _v53mp_root(tmp_path)
    tables, split = _stub_data(root)
    args = [
        "--output-root", str(root), "plan-init",
        "--tier", "2k", "--tables-dir", str(tables), "--split", str(split),
        "--config", str(CONFIG),
    ]
    first = _run(args)
    assert first.returncode == 0, first.stderr
    second = _run(args)
    assert second.returncode != 0
    assert "PLAN_EXISTS" in second.stderr


# --------------------------------------------------------------------------
# gate-check (S0-F readiness)
# --------------------------------------------------------------------------


def test_gate_check_fails_closed_on_missing_data(tmp_path: Path) -> None:
    root = _v53mp_root(tmp_path)
    empty_tables = tmp_path / "empty_tables"
    empty_tables.mkdir()
    proc = _run([
        "gate-check",
        "--config", str(CONFIG),
        "--tables-dir", str(empty_tables),
        "--split", str(empty_tables / "split_manifest_v53mp.json"),
        "--truth-log", str(empty_tables / "truth_access_log.jsonl"),
    ])
    assert proc.returncode != 0, proc.stdout
    report = json.loads(proc.stdout)
    assert report["ok"] is False
    items = report["items"]
    # config + driver + export are Lane-D-owned and must pass
    assert items["config_yaml"] == "pass"
    assert items["run_driver_r3_v53mp"] == "pass"
    assert items["export_move_process_bundle"] == "pass"
    # every data gate fails closed on the empty tree
    for name in (
        "data_tables_six", "data_manifest", "teacher_manifest",
        "split_manifest", "truth_access_log",
    ):
        assert items[name] == "fail", name


def test_gate_check_fails_closed_on_missing_config(tmp_path: Path) -> None:
    root = _v53mp_root(tmp_path)
    tables, split = _stub_data(root)
    proc = _run([
        "gate-check",
        "--config", str(tmp_path / "missing_config.yaml"),
        "--tables-dir", str(tables),
        "--split", str(split),
    ])
    assert proc.returncode != 0
    report = json.loads(proc.stdout)
    assert report["items"]["config_yaml"] == "fail"


def test_gate_check_full_pass(tmp_path: Path) -> None:
    """Passes once the parallel lanes + torch environment exist.

    TODO: loss/trainer import items need torch (module-level import in
    key_atom_loss_v53mp_v3 / key_atom_v53mp_train_v3); the loader and
    endpoint-spec items need Lane-A/B symbols.  Skip (never silently pass)
    until then — on the training host this test must run green with 7/7.
    """
    loss = pytest.importorskip(
        "pes2ts_core.pesn.models.key_atom_loss_v53mp_v3",
        reason="TODO: Lane B loss module or torch not available yet",
    )
    assert hasattr(loss, "key_atom_loss_v53mp")
    trainer = pytest.importorskip(
        "pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3",
        reason="TODO: Lane C trainer module or torch not available yet",
    )
    assert hasattr(trainer, "train_run_v53mp")
    loader = pytest.importorskip(
        "pes2ts_core.pesn.offline.data.key_atom_dataset_v3"
    )
    if not hasattr(loader, "ShardedKeyAtomDatasetV3"):
        pytest.skip("TODO: ShardedKeyAtomDatasetV3 (sharded loader) not landed")
    labels = pytest.importorskip(
        "pes2ts_core.pesn.offline.data.key_atom_endpoint_labels_v1"
    )
    if not hasattr(labels, "build_endpoint_labels_spec_v1"):
        pytest.skip("TODO: endpoint_label_spec_v1 generator not landed")
    publish = pytest.importorskip("pes2ts_core.pesn.offline.publishing.export")
    assert hasattr(publish, "export_move_process_bundle")

    root = _v53mp_root(tmp_path)
    tables, split = _stub_data(root)
    proc = _run([
        "gate-check",
        "--config", str(CONFIG),
        "--tables-dir", str(tables),
        "--split", str(split),
    ])
    assert proc.returncode == 0, proc.stdout + proc.stderr
    report = json.loads(proc.stdout)
    assert report["ok"] is True
    assert all(v == "pass" for v in report["items"].values()), report["items"]


# --------------------------------------------------------------------------
# run: authorization gate
# --------------------------------------------------------------------------


def test_run_refuses_without_s1_authorization(tmp_path: Path) -> None:
    root = _v53mp_root(tmp_path)
    tables, split = _stub_data(root)
    plan = _run([
        "--output-root", str(root), "plan-init",
        "--tier", "2k", "--tables-dir", str(tables), "--split", str(split),
        "--config", str(CONFIG),
    ])
    assert plan.returncode == 0, plan.stderr

    gpu_lock = tmp_path / "gpu.lock"
    ledger = root / "run_records.jsonl"
    proc = _run([
        "--output-root", str(root), "run",
        "--run-id", "v53mp_2k_A_s17",
        "--tables-dir", str(tables), "--split", str(split),
        "--config", str(CONFIG),
        "--ledger", str(ledger),
        "--gpu-lock", str(gpu_lock),
    ])
    assert proc.returncode == 1, proc.stdout + proc.stderr

    # refusal JSON persisted + printed
    refusal = json.loads((root / "refusals" / "v53mp_2k_A_s17.json").read_text())
    assert refusal["action"] == "refused"
    assert refusal["reason"] == "authorization"
    assert refusal["authorization_bit"] == "s1_authorized"
    assert refusal["value"] is False
    assert json.loads(proc.stdout)["reason"] == "authorization"

    # fail-closed: NO GPU touch, NO ledger mutation
    assert not gpu_lock.exists()
    assert not ledger.exists()


def test_run_refuses_unknown_run_id(tmp_path: Path) -> None:
    root = _v53mp_root(tmp_path)
    tables, split = _stub_data(root)
    plan = _run([
        "--output-root", str(root), "plan-init",
        "--tier", "2k", "--tables-dir", str(tables), "--split", str(split),
        "--config", str(CONFIG),
    ])
    assert plan.returncode == 0, plan.stderr
    proc = _run([
        "--output-root", str(root), "run",
        "--run-id", "v53mp_2k_B_s99",   # not in plan
        "--tables-dir", str(tables), "--split", str(split),
        "--config", str(CONFIG),
    ])
    assert proc.returncode == 2
    assert "RUN_NOT_IN_PLAN" in proc.stderr
