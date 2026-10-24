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
    # the authorization gate must read a config with s1_authorized=false —
    # never the live campaign config (whose bit flips as stages advance)
    import yaml
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg["governance"]["authorization"]["s1_authorized"] = False
    cfg_path = tmp_path / "config_unauthorized.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
                        encoding="utf-8")
    plan = _run([
        "--output-root", str(root), "plan-init",
        "--tier", "2k", "--tables-dir", str(tables), "--split", str(split),
        "--config", str(cfg_path),
    ])
    assert plan.returncode == 0, plan.stderr

    gpu_lock = tmp_path / "gpu.lock"
    ledger = root / "run_records.jsonl"
    proc = _run([
        "--output-root", str(root), "run",
        "--run-id", "v53mp_2k_A_s17",
        "--tables-dir", str(tables), "--split", str(split),
        "--config", str(cfg_path),
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


# --------------------------------------------------------------------------
# run: --cuda-memory-fraction CLI (parse + 0<f<=1 validation + pass-through)
# --------------------------------------------------------------------------


def _driver_module():
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from scripts.campaigns import r3_v53mp_run as drv

    return drv


def test_run_cli_cuda_memory_fraction_default_none() -> None:
    drv = _driver_module()
    args = drv._build_parser().parse_args(
        ["run", "--run-id", "v53mp_2k_A_s17"])
    # default None => zero behaviour change (no fraction applied)
    assert args.cuda_memory_fraction is None


def test_run_cli_cuda_memory_fraction_parses_and_validates() -> None:
    drv = _driver_module()
    parser = drv._build_parser()
    # valid 0 < f <= 1 parses to float
    args = parser.parse_args([
        "run", "--run-id", "v53mp_2k_A_s17",
        "--cuda-memory-fraction", "0.17"])
    assert args.cuda_memory_fraction == 0.17
    args = parser.parse_args([
        "run", "--run-id", "v53mp_2k_A_s17",
        "--cuda-memory-fraction", "1.0"])
    assert args.cuda_memory_fraction == 1.0
    # out-of-range / unparseable values exit 2 (argparse fail-closed)
    for bad in ("0", "-0.1", "1.5", "nan", "abc"):
        with pytest.raises(SystemExit) as exc:
            parser.parse_args([
                "run", "--run-id", "v53mp_2k_A_s17",
                "--cuda-memory-fraction", bad])
        assert exc.value.code == 2, bad


def test_run_cli_cuda_memory_fraction_rejects_out_of_range_subprocess() -> None:
    # end-to-end CLI surface: refused by argparse BEFORE any ledger/GPU work
    proc = _run([
        "run", "--run-id", "v53mp_2k_A_s17",
        "--cuda-memory-fraction", "1.5",
    ])
    assert proc.returncode == 2
    assert "cuda-memory-fraction" in proc.stderr


def test_run_cli_cuda_memory_fraction_passes_into_trainer_config() -> None:
    """cmd_run must hand args.cuda_memory_fraction to V53MPTrainerConfig."""
    import inspect

    drv = _driver_module()
    source = inspect.getsource(drv.cmd_run)
    assert "cuda_memory_fraction=args.cuda_memory_fraction" in source


# --------------------------------------------------------------------------
# run: --dev-pack auto binding discovery (migration *.sha256sidecar.json)
# --------------------------------------------------------------------------


def _make_candidate(tmp_path: Path) -> Path:
    candidate = tmp_path / "dev_pack"
    candidate.mkdir()
    return candidate


def _touch_sidecars(candidate: Path, n: int = 4) -> None:
    for i in range(n):
        (candidate / "blobs.npy.{}.sha256sidecar.json".format(i)).write_text(
            "{}", encoding="utf-8")


def test_dev_pack_binding_selected_despite_sidecar_jsons(tmp_path: Path) -> None:
    """1 binding json + N migration sidecars: the binding is selected."""
    drv = _driver_module()
    candidate = _make_candidate(tmp_path)
    binding = candidate / ("b" * 64 + ".json")
    binding.write_text("{}", encoding="utf-8")
    _touch_sidecars(candidate)
    assert drv._resolve_dev_pack_binding(candidate) == binding
    assert drv._dev_pack_binding_jsons(candidate) == [binding]


def test_dev_pack_sidecars_without_binding_still_refused(
        tmp_path: Path) -> None:
    """Sidecars but no binding json: DEV_PACK_MISSING, found 0."""
    drv = _driver_module()
    candidate = _make_candidate(tmp_path)
    _touch_sidecars(candidate)
    with pytest.raises(drv.V53MPConfigError) as exc:
        drv._resolve_dev_pack_binding(candidate)
    assert "DEV_PACK_MISSING" in str(exc.value)
    assert "found 0" in str(exc.value)


def test_dev_pack_two_bindings_still_refused(tmp_path: Path) -> None:
    """Two real binding jsons: DEV_PACK_MISSING, found 2 (fail-closed)."""
    drv = _driver_module()
    candidate = _make_candidate(tmp_path)
    (candidate / ("a" * 64 + ".json")).write_text("{}", encoding="utf-8")
    (candidate / ("b" * 64 + ".json")).write_text("{}", encoding="utf-8")
    _touch_sidecars(candidate)
    with pytest.raises(drv.V53MPConfigError) as exc:
        drv._resolve_dev_pack_binding(candidate)
    assert "DEV_PACK_MISSING" in str(exc.value)
    assert "found 2" in str(exc.value)


# --------------------------------------------------------------------------
# gate-check: manifest sha verification via migration sidecars (no 54 GB
# re-read — the gate-check subprocess runs OUTSIDE the training cgroup and
# must not re-accumulate the full data set in the guest page cache)
# --------------------------------------------------------------------------


def _write_all_sidecars(tables: Path, names=SIX_TABLES) -> None:
    """Populate migration sidecars for the listed stub files."""
    from scripts.infra.sha_sidecar import sidecar_sha256

    for name in names:
        sidecar_sha256(tables / name)


def _spy_rehash(drv, monkeypatch, record: list) -> None:
    """Patch the rehash path to record calls but keep true digests."""
    import hashlib as _hashlib
    from pathlib import Path as _Path

    def _fake_sha256_file(path):
        record.append(str(path))
        return _hashlib.sha256(_Path(path).read_bytes()).hexdigest()

    monkeypatch.setattr(drv.sha_sidecar, "sha256_file", _fake_sha256_file)


def test_verify_manifest_shas_sidecar_hit_never_reads_data_file(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Valid sidecars for every listed file: NO data-file read at all."""
    drv = _driver_module()
    root = _v53mp_root(tmp_path)
    tables, _split = _stub_data(root)
    _write_all_sidecars(tables)

    def _forbidden(path):
        pytest.fail("cache hit must not rehash {}".format(path))

    monkeypatch.setattr(drv.sha_sidecar, "sha256_file", _forbidden)

    result, detail = drv._verify_manifest_shas(tables, "data_manifest.json")
    assert result == "pass", detail
    assert "{} verified ({} file(s))".format(
        "data_manifest.json", len(SIX_TABLES)) in detail


def test_verify_manifest_shas_symlink_uses_target_sidecar(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Migrated source symlink resolves to the TARGET-side sidecar."""
    drv = _driver_module()
    root = _v53mp_root(tmp_path)
    tables, _split = _stub_data(root)
    _write_all_sidecars(tables)

    # migrate one table: real file moves to the migrated volume, the
    # manifest-listed path becomes a symlink; its tables-side sidecar goes
    # away with the move — only the TARGET-side sidecar remains.
    real_dir = tmp_path / "migrated"
    real_dir.mkdir()
    real = real_dir / "irc_curve.csv"
    real.write_bytes((tables / "irc_curve.csv").read_bytes())
    tables.joinpath("irc_curve.csv").unlink()
    tables.joinpath("irc_curve.csv").symlink_to(real)
    from scripts.infra.sha_sidecar import sidecar_path, sidecar_sha256

    sidecar_path(tables / "irc_curve.csv").unlink()  # link-side sidecar gone
    sidecar_sha256(real)  # migration wrote the sidecar NEXT TO THE TARGET
    assert not sidecar_path(tables / "irc_curve.csv").exists()
    assert sidecar_path(real).is_file()

    def _forbidden(path):
        pytest.fail("target-side sidecar hit must not rehash {}".format(path))

    monkeypatch.setattr(drv.sha_sidecar, "sha256_file", _forbidden)

    result, detail = drv._verify_manifest_shas(tables, "data_manifest.json")
    assert result == "pass", detail


def test_verify_manifest_shas_cache_miss_rehashes_and_drops(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No sidecars: every file rehashes, then drops its page cache; the
    refreshed sidecars preserve the ORIGINAL digest semantics."""
    drv = _driver_module()
    root = _v53mp_root(tmp_path)
    tables, _split = _stub_data(root)  # no sidecars -> all misses

    rehashed: list[str] = []
    dropped: list[str] = []
    _spy_rehash(drv, monkeypatch, rehashed)
    monkeypatch.setattr(drv, "_drop_page_cache",
                        lambda p: dropped.append(str(p)))

    result, detail = drv._verify_manifest_shas(tables, "data_manifest.json")
    assert result == "pass", detail
    assert sorted(Path(p).name for p in rehashed) == sorted(SIX_TABLES)
    # every rehashed file also dropped its cache (never accumulate)
    assert sorted(dropped) == sorted(rehashed)
    # original sidecar semantics: sidecars now exist AND carry the true digests
    from scripts.infra.sha_sidecar import sidecar_path

    for name in SIX_TABLES:
        side = sidecar_path(tables / name)
        assert side.is_file(), name
        doc = json.loads(side.read_text(encoding="utf-8"))
        assert doc["sha256"] == hashlib.sha256(
            (tables / name).read_bytes()).hexdigest()


def test_verify_manifest_shas_mismatch_still_fails(
        tmp_path: Path) -> None:
    """Cache-miss rehash whose digest drifts from the manifest still fails."""
    drv = _driver_module()
    root = _v53mp_root(tmp_path)
    tables, _split = _stub_data(root)
    (tables / "irc_curve.csv").write_text(
        "curve_id\ntampered\n", encoding="utf-8")

    result, detail = drv._verify_manifest_shas(tables, "data_manifest.json")
    assert result == "fail"
    assert "sha mismatch" in detail


def test_verify_manifest_shas_sidecar_hit_mismatch_still_fails(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A cache hit cannot mask drift: tampered file + matching sidecar but
    manifest sha frozen -> still fail (fail-closed, no rehash needed)."""
    drv = _driver_module()
    root = _v53mp_root(tmp_path)
    tables, _split = _stub_data(root)
    _write_all_sidecars(tables)  # all six files cached ...
    (tables / "irc_curve.csv").write_text(
        "curve_id\ntampered\n", encoding="utf-8")  # ... then drift happens
    from scripts.infra.sha_sidecar import sidecar_sha256

    sidecar_sha256(tables / "irc_curve.csv")  # cache matches the tampering

    def _forbidden(path):
        pytest.fail("cache hit must not rehash {}".format(path))

    monkeypatch.setattr(drv.sha_sidecar, "sha256_file", _forbidden)

    result, detail = drv._verify_manifest_shas(tables, "data_manifest.json")
    assert result == "fail"
    assert "sha mismatch" in detail


def test_verify_manifest_shas_corrupt_sidecar_falls_back_to_rehash(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A corrupt sidecar is a cache miss: full rehash, sidecar rewritten."""
    drv = _driver_module()
    root = _v53mp_root(tmp_path)
    tables, _split = _stub_data(root)
    _write_all_sidecars(tables)
    from scripts.infra.sha_sidecar import sidecar_path

    bad = sidecar_path(tables / "irc_curve.csv")
    bad.write_text("{corrupt", encoding="utf-8")

    rehashed: list[str] = []
    _spy_rehash(drv, monkeypatch, rehashed)
    monkeypatch.setattr(drv, "_drop_page_cache", lambda p: None)

    result, detail = drv._verify_manifest_shas(tables, "data_manifest.json")
    assert result == "pass", detail
    # ONLY the corrupt-sidecar file rehashes; the other five stay cached
    assert [Path(p).name for p in rehashed] == ["irc_curve.csv"]
    doc = json.loads(bad.read_text(encoding="utf-8"))
    assert doc["sha256"] == hashlib.sha256(
        (tables / "irc_curve.csv").read_bytes()).hexdigest()


def test_drop_page_cache_never_raises(tmp_path: Path) -> None:
    """_drop_page_cache is advisory: valid/missing/foreign fds never raise."""
    drv = _driver_module()
    target = tmp_path / "x.bin"
    target.write_bytes(b"x" * 4096)
    drv._drop_page_cache(target)  # Linux: posix_fadvise; elsewhere no-op
    drv._drop_page_cache(tmp_path / "missing.bin")  # missing: tolerated
