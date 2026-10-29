"""V5.4.1 WP4c runner acceptance (``scripts/campaigns/v541_run.py``).

Scope (plan §12 / §14 / §17): the INDEPENDENT V5.4.1 campaign runner —
CLI surface, plan/pin freeze + tamper refusal, typed refusals for
unimplemented registry ids, mode labels (smoke / profile / formal),
fail-closed formal gates, append-only V541 run records and the
"legacy ledger is never written" guarantee.

Everything runs on stub tables + the tiny synthetic six-table fixture from
``tests/pesn/test_r3_v53mp_trainer`` — no real 50K data, no real dev/confirm
access, no GPU, no training beyond a couple of CPU fixture updates.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.agent_ops.v541_config_schema import load_config  # noqa: E402
from scripts.campaigns import v541_run as R  # noqa: E402
from scripts.campaigns.v541_provenance import (  # noqa: E402
    source_manifest,
)
from scripts.campaigns.v541_runplan import (  # noqa: E402
    classify_started_runs,
    plan_paths,
    read_plan,
    started_run_ids,
    verify_plan_pin_v541,
    write_plan,
)

CONFIG = REPO / "config" / "campaigns" / "pes2ts_v541_embed_50k.yaml"
SIX_TABLES = (
    "irc_curve.csv",
    "irc_node.csv",
    "irc_edge.csv",
    "reaction_intent.csv",
    "curve_quality.csv",
    "sample_view.csv",
)
POOL_CURVES = ("cv0", "cv1", "cv2")
DEV_CURVES = ("cv_dev0", "cv_dev1")


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
def _root(tmp_path: Path) -> Path:
    root = tmp_path / "outputs" / "pesn" / "v541_embed_50k"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _stub_tables(tmp_path: Path) -> tuple[Path, Path]:
    tables = tmp_path / "data"
    tables.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}
    for name in SIX_TABLES:
        (tables / name).write_text("curve_id\n", encoding="utf-8")
        files[name] = hashlib.sha256((tables / name).read_bytes()).hexdigest()
    (tables / "data_manifest.json").write_text(
        json.dumps({"files": [{"path": n, "sha256": files[n]}
                              for n in SIX_TABLES]}, sort_keys=True),
        encoding="utf-8")
    (tables / "teacher_manifest.json").write_text(
        json.dumps({"files": []}, sort_keys=True), encoding="utf-8")
    split = tables / "split_manifest_v53mp.json"
    split.write_text(
        json.dumps({
            "schema": "v53mp_split_v1",
            "subsets": {
                "pool": [{"curve_id": c, "rank": 100} for c in POOL_CURVES],
                "dev": list(DEV_CURVES),
                "confirm": [],
            },
        }, sort_keys=True),
        encoding="utf-8")
    (tables / "truth_access_log.jsonl").write_text(
        '{"source": "stub", "purpose": "test"}\n', encoding="utf-8")
    return tables, split


def _make_plan(root: Path, tables: Path, split: Path, *,
               config_path: Path | None = None,
               cfg: object | None = None) -> dict:
    return write_plan(
        root, cfg if cfg is not None else load_config(),
        config_path=str(config_path or CONFIG), tables_dir=tables,
        split_path=split)


def _null_budget_config(tmp_path: Path) -> Path:
    """A TEMPORARY config copy whose WP5 budget is back to pre-signoff.

    The frozen repo config now carries the signed W0/W1 profiling budget, so
    every "a null budget blocks T1" expectation must be built on this copy —
    never assumed from the real file.  Nothing under ``outputs/`` or the
    repo config is touched.
    """
    import copy

    import yaml

    cfg = copy.deepcopy(yaml.safe_load(CONFIG.read_text(encoding="utf-8")))
    cfg["budget"]["per_candidate_core_time"] = None
    cfg["budget"]["per_candidate_status"] = "pending_wp5_profiling"
    path = tmp_path / "v541_null_budget.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path


def _cli(root: Path, *args: str) -> int:
    return R.main(["--output-root", str(root), *args])


def _records(root: Path) -> list[dict]:
    path = root / "run_records.jsonl"
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in
            path.read_text(encoding="utf-8").splitlines() if ln.strip()]


@pytest.fixture()
def plan_root(tmp_path: Path) -> tuple[Path, Path, Path]:
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    _make_plan(root, tables, split)
    return root, tables, split


@pytest.fixture()
def fixture_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Patch the 50K loader with the tiny synthetic six-table fixture."""
    from tests.pesn.test_r3_v53mp_trainer import DEFAULT_FOLDS, _build_env

    env = _build_env(tmp_path / "fixture", DEFAULT_FOLDS)
    inputs = R.RunInputs(
        dataset=env.ds,
        labels=env.labels,
        endpoint=env.endpoint,
        split_doc={"schema": "v53mp_split_v1",
                   "subsets": {"pool": list(POOL_CURVES),
                               "dev": list(DEV_CURVES),
                               "confirm": []}},
        pool_curve_ids=list(POOL_CURVES),
        dev_curve_ids=list(DEV_CURVES),
        confirm_curve_ids=[],
        provenance={"tier": "50k", "shard_path": "fixture",
                    "split_manifest_sha256": "s" * 64,
                    "data_manifest_sha256": "d" * 64,
                    "teacher_manifest_sha256": "t" * 64},
    )
    monkeypatch.setattr(R, "load_run_inputs", lambda **kwargs: inputs)
    return env, inputs


def _run_args(tables: Path, lock: Path) -> list[str]:
    return ["run",
            "--run-id", "V541_E0_S17", "--embedding", "E0", "--seed", "17",
            "--tables-dir", str(tables), "--device", "cpu",
            "--feature-workers", "0", "--gpu-lock", str(lock)]


# ---------------------------------------------------------------------------
# CLI: validate-config
# ---------------------------------------------------------------------------
def test_validate_config_prereg_exits_zero(tmp_path: Path) -> None:
    root = _root(tmp_path)
    assert _cli(root, "validate-config", "--config", str(CONFIG)) == 0


def test_validate_config_formal_lists_t1_gate_blocks(tmp_path: Path,
                                                     capsys) -> None:
    root = _root(tmp_path)
    code = _cli(root, "validate-config", "--formal", "--config", str(CONFIG))
    report = json.loads(capsys.readouterr().out)
    assert code == 1, report
    assert report["ok"] is False
    assert report["mode"] == "formal"
    codes = {b["code"] for b in report["t1_blocks"]}
    # formal must check the GATES, not just the (signed) thresholds — and
    # the frozen config is now signed at the CONFIG level too (thresholds
    # 2026-10-27, WP5 budget), so no V541_T1_CONFIG blocker remains here.
    assert "V541_T1_CONFIG" not in codes
    assert report["t1_blocked"] == []
    assert "V541_GATE_H0" in codes
    assert "V541_GATE_H1" in codes
    assert "V541_GATE_SMOKE_200" in codes
    assert "V541_GATE_RUNPLAN_PIN" in codes
    # no instantiated plan/budget in this root -> the signoff gate blocks
    assert "V541_GATE_BUDGET_SIGNOFF" in codes
    assert report["t1_ready"] is False


def test_null_budget_config_blocks_formal_t1(tmp_path: Path,
                                             capsys) -> None:
    """Explicit NULL-budget copy still blocks T1 (never the real config)."""
    root = _root(tmp_path)
    null_cfg = _null_budget_config(tmp_path)
    code = _cli(root, "validate-config", "--formal", "--config", str(null_cfg))
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["t1_ready"] is False
    config_blocks = [b for b in report["t1_blocks"]
                     if b["code"] == "V541_T1_CONFIG"]
    assert config_blocks, report["t1_blocks"]
    assert "per_candidate_core_time" in config_blocks[0]["detail"]


def test_validate_config_rejects_sampler_drift(tmp_path: Path,
                                               capsys) -> None:
    root = _root(tmp_path)
    import copy

    import yaml

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(cfg)
    cfg["training_contract"]["sampler_binding"]["spec_sha256"] = "0" * 64
    bad = tmp_path / "drift.yaml"
    bad.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    code = _cli(root, "validate-config", "--config", str(bad))
    payload = json.loads(capsys.readouterr().out)
    assert code == 2
    assert payload["code"] == "V541_SAMPLER_DRIFT"


# ---------------------------------------------------------------------------
# CLI: audit-data-and-manifests
# ---------------------------------------------------------------------------
def test_audit_fails_closed_when_data_missing(tmp_path: Path,
                                              capsys) -> None:
    root = _root(tmp_path)
    empty = tmp_path / "nope"
    empty.mkdir()
    code = _cli(root, "audit-data-and-manifests", "--tables-dir", str(empty))
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["ok"] is False
    assert "data_manifest_present" in report["failed"]
    assert "split_manifest_present" in report["failed"]
    # missing data NEVER passes
    assert report["items"]["data_manifest_present"]["status"] == "fail"
    assert report["confirm_read"] is False
    assert report["new_qc_calls"] == 0
    assert (root / "audits" / "data_audit.json").is_file()


def test_audit_on_fixture_pool_split_strata(tmp_path: Path,
                                            fixture_inputs,
                                            capsys) -> None:
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    cfg = load_config()
    report = R.audit_data_and_manifests(
        config=cfg, config_path=str(CONFIG), tables_dir=tables,
        split_path=split, inputs=fixture_inputs[1])
    assert report["ok"] is True, report["failed"]
    items = report["items"]
    assert items["subset_disjointness"]["status"] == "pass"
    assert items["confirm_excluded_from_pool"]["status"] == "pass"
    assert items["direction_coverage"]["status"] == "pass"
    assert items["strata_pool_sha256"]["status"] == "pass"
    assert items["strata_coverage"]["status"] == "pass"
    assert items["dev_exclusion_only"]["status"] == "pass"
    for seed in (17, 29, 43):
        assert items["inner_split_s{}".format(seed)]["status"] == "pass"
    counts = report["counts"]
    assert counts["endpoint_b_index_zero_quirks"] >= 0
    assert len(counts["strata_pool_sha256"]) == 64
    assert counts["shas"]["r3_init_sha256"] == (
        cfg["training_contract"]["backbone_init"]["sha256"])
    assert report["dev_used_for"] == "curve_exclusion_only"


def test_audit_never_reads_confirm(tmp_path: Path, fixture_inputs) -> None:
    """A confirm curve in the pool is a typed FAIL, never a silent pass."""
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    inputs = fixture_inputs[1]
    inputs.confirm_curve_ids = ["cv0"]          # poisoned pool
    report = R.audit_data_and_manifests(
        config=load_config(), config_path=str(CONFIG), tables_dir=tables,
        split_path=split, inputs=inputs)
    assert report["ok"] is False
    assert report["items"]["confirm_curve_not_in_pool"]["status"] == "fail"
    assert report["items"]["inner_split_s17"]["status"] == "fail"


# ---------------------------------------------------------------------------
# plan-init / pin
# ---------------------------------------------------------------------------
def test_plan_init_creates_four_artifacts(tmp_path: Path) -> None:
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    code = _cli(root, "plan-init", "--config", str(CONFIG),
                "--tables-dir", str(tables), "--split", str(split))
    assert code == 0
    paths = plan_paths(root)
    for key in ("plan", "pin", "signoff", "budget"):
        assert paths[key].is_file(), key

    plan = read_plan(paths["plan"])
    assert len(plan["run_ids"]) == 20
    assert plan["header"]["campaign_mode"] == "full"
    assert plan["header"]["objective"] == "unified_move_process"
    verified = verify_plan_pin_v541(paths["plan"], paths["pin"])
    assert len(verified["code_sha256"]) == 64

    # the three run classes + T3 shape survive instantiation
    runs = plan["lines"][1:]
    by_class: dict[str, list] = {}
    for line in runs:
        by_class.setdefault(line["class"], []).append(line["run_id"])
    assert set(by_class) == {"nominal", "conditional", "diagnostic"}
    t3 = [r for r in runs if r["stage"] == "T3"]
    assert len(t3) == 10
    assert all(r["unactivated_status"] == "gate_skipped" for r in t3)
    assert all(r["condition"] == "activated_only_by_inner_val_top2" for r in t3)
    t0r = [r for r in runs if r["stage"] == "T0-R"]
    assert {r["embedding_id"] for r in t0r} == {"E0"}
    assert {r["seed"] for r in t0r} == {17, 29, 43}
    e5 = [r for r in runs if r["embedding_id"] == "E5" and r["stage"] == "T2"]
    assert e5 and e5[0]["condition"].startswith("parents_E1_E4")
    e5_t3 = [r for r in runs if r["embedding_id"] == "E5" and r["stage"] == "T3"]
    assert all(r["parent_gate"].startswith("parents_E1_E4") for r in e5_t3)
    t4 = [r for r in runs if r["stage"] == "T4"]
    assert len(t4) == 2 and all(r["class"] == "diagnostic" for r in t4)


def test_plan_lines_carry_every_required_sha(tmp_path: Path) -> None:
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    _make_plan(root, tables, split)
    plan = read_plan(plan_paths(root)["plan"])
    required = {
        "code_sha256", "environment_sha256", "data_manifest_sha256",
        "teacher_manifest_sha256", "split_sha256", "label_spec_sha256",
        "r3_init_sha256", "common_protocol_sha256", "embedding_spec_sha256",
        "feature_recipe_sha256", "config_sha256", "budget_sha256",
    }
    for line in plan["lines"][1:]:
        missing = [k for k in required
                   if not isinstance(line.get(k), str) or len(line[k]) != 64]
        assert not missing, (line["run_id"], missing)
        assert line["objective"] == "unified_move_process"
        assert line["status"] == "planned" and line["attempt"] == 0
        assert line["confirm"] is False and line["new_qc_calls"] == 0


def test_plan_common_sha_shared_but_recipe_shas_per_candidate(
        tmp_path: Path) -> None:
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    _make_plan(root, tables, split)
    runs = read_plan(plan_paths(root)["plan"])["lines"][1:]
    common = {r["common_protocol_sha256"] for r in runs}
    assert len(common) == 1, "one numeric protocol => one common SHA"
    by_emb: dict[str, set] = {}
    for r in runs:
        by_emb.setdefault(r["embedding_id"], set()).add(
            (r["embedding_spec_sha256"], r["feature_recipe_sha256"]))
    # every candidate has its OWN recipe digests (never one shared value)
    assert all(len(v) == 1 for v in by_emb.values())
    all_pairs = {pair for pairs in by_emb.values() for pair in pairs}
    assert len({p[0] for p in all_pairs}) == len(by_emb)
    assert len({p[1] for p in all_pairs}) == len(by_emb)


def test_plan_init_refuses_existing_pin_without_force(tmp_path: Path) -> None:
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    _make_plan(root, tables, split)
    with pytest.raises(Exception) as exc:
        _make_plan(root, tables, split)
    assert getattr(exc.value, "code", "") == "V541_PLAN_PIN_EXISTS"


def test_plan_init_force_refused_once_a_run_started(tmp_path: Path) -> None:
    """A ledger attempt WITHOUT a ``mode`` (legacy) is conservatively blocking."""
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    _make_plan(root, tables, split)
    (root / "run_records.jsonl").write_text(
        json.dumps({"event": "attempt_start", "run_id": "V541_E0_S17"}) + "\n",
        encoding="utf-8")
    assert started_run_ids(root) == ["V541_E0_S17"]
    classified = classify_started_runs(root)
    assert list(classified["technical"]) == []
    assert classified["blocking"]["V541_E0_S17"]["mode"] == "unknown"
    with pytest.raises(Exception) as exc:
        write_plan(root, load_config(), config_path=str(CONFIG),
                   tables_dir=tables, split_path=split, force=True)
    assert getattr(exc.value, "code", "") == "V541_PLAN_STARTED_RUNS"


# ---------------------------------------------------------------------------
# plan re-freeze: technical evidence survives, formal evidence blocks
# (plan §8.2 — smoke/profile are NON-formal; WP5 back-fills the budget and
# the plan is then re-frozen with the new formal code/common/plan SHAs)
# ---------------------------------------------------------------------------
def _technical_evidence(root: Path) -> dict:
    """Smoke + profile artifacts and their (mode-labelled) ledger attempts."""
    smoke = root / "smoke" / "V541_E0_S17"
    smoke.mkdir(parents=True, exist_ok=True)
    (smoke / "metrics.json").write_text(
        json.dumps({"is_formal_run": False, "run_outcome": "done"}),
        encoding="utf-8")
    (smoke / "manifest.jsonl").write_text(
        '{"event": "checkpoint_last", "update": 200}\n', encoding="utf-8")
    profile = root / "profile" / "V541_E1_S17"
    profile.mkdir(parents=True, exist_ok=True)
    (profile / "profile.json").write_text(
        json.dumps({"profile_only": True, "is_formal_run": False}),
        encoding="utf-8")
    records = root / "run_records.jsonl"
    records.write_text(
        json.dumps({"event": "attempt_start", "run_id": "V541_E0_S17",
                    "mode": "smoke"}) + "\n"
        + json.dumps({"event": "attempt_end", "run_id": "V541_E0_S17",
                      "mode": "smoke", "outcome": "done"}) + "\n"
        + json.dumps({"event": "attempt_start", "run_id": "V541_E1_S17",
                      "mode": "profile"}) + "\n",
        encoding="utf-8")
    return {"smoke": smoke, "profile": profile, "records": records}


def test_force_refreeze_preserves_technical_evidence(tmp_path: Path,
                                                     capsys) -> None:
    """Only smoke/profile evidence -> ``--force`` re-pins, history intact."""
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    _make_plan(root, tables, split)
    paths = plan_paths(root)
    old_pin = json.loads(paths["pin"].read_text(encoding="utf-8"))

    evidence = _technical_evidence(root)
    before_smoke = {p: p.read_bytes() for p in sorted(evidence["smoke"].iterdir())}
    before_profile = {p: p.read_bytes()
                      for p in sorted(evidence["profile"].iterdir())}
    before_records = evidence["records"].read_bytes()

    classified = classify_started_runs(root)
    assert sorted(classified["technical"]) == ["V541_E0_S17", "V541_E1_S17"]
    assert classified["blocking"] == {}
    assert started_run_ids(root) == ["V541_E0_S17", "V541_E1_S17"]

    code = _cli(root, "plan-init", "--force", "--config", str(CONFIG),
                "--tables-dir", str(tables), "--split", str(split))
    payload = json.loads(capsys.readouterr().out)
    assert code == 0, payload

    refreeze = payload["refreeze"]
    assert refreeze["forced"] is True and refreeze["rebuilt"] is True
    assert refreeze["preserved_technical_runs"] == ["V541_E0_S17",
                                                    "V541_E1_S17"]
    for key in ("new_formal_code_sha256",
                "new_formal_common_protocol_sha256",
                "new_formal_plan_sha256", "new_formal_budget_sha256",
                "new_formal_environment_sha256"):
        assert len(refreeze[key]) == 64, key
    assert refreeze["new_formal_plan_sha256"] == payload["plan_sha256"]
    assert refreeze["new_formal_code_sha256"] == payload["code_sha256"]
    assert refreeze["new_formal_common_protocol_sha256"] == \
        payload["common_protocol_sha256"]

    # --- the technical history is preserved BYTE-FOR-BYTE -----------------
    assert {p: p.read_bytes() for p in sorted(evidence["smoke"].iterdir())} \
        == before_smoke
    assert {p: p.read_bytes() for p in sorted(evidence["profile"].iterdir())} \
        == before_profile
    assert evidence["records"].read_bytes() == before_records

    # --- the new pin really binds the NEW formal SHAs ---------------------
    new_pin = json.loads(paths["pin"].read_text(encoding="utf-8"))
    assert new_pin["sha256"] != old_pin["sha256"]
    assert new_pin["sha256"] == refreeze["new_formal_plan_sha256"]
    assert new_pin["code_sha256"] == refreeze["new_formal_code_sha256"]
    assert new_pin["common_protocol_sha256"] == \
        refreeze["new_formal_common_protocol_sha256"]
    assert new_pin["budget_sha256"] == refreeze["new_formal_budget_sha256"]
    # ...and the rebuilt plan still verifies against it
    verified = verify_plan_pin_v541(paths["plan"], paths["pin"])
    assert verified["code_sha256"] == refreeze["new_formal_code_sha256"]


def test_force_refreeze_blocked_by_formal_or_diagnostic_dirs(
        tmp_path: Path) -> None:
    """A formal / T4 run DIRECTORY alone keeps refusing the rebuild."""
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    _make_plan(root, tables, split)
    paths = plan_paths(root)
    old_pin_sha = json.loads(paths["pin"].read_text(encoding="utf-8"))["sha256"]

    for subdir in ("runs", "diagnostics"):
        run_dir = root / subdir / "V541_E1_S17"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "metrics.json").write_text("{}", encoding="utf-8")
    classified = classify_started_runs(root)
    assert sorted(classified["blocking"]) == ["V541_E1_S17"]

    with pytest.raises(Exception) as exc:
        write_plan(root, load_config(), config_path=str(CONFIG),
                   tables_dir=tables, split_path=split, force=True)
    assert getattr(exc.value, "code", "") == "V541_PLAN_STARTED_RUNS"
    # nothing was deleted / overwritten
    assert (root / "runs" / "V541_E1_S17" / "metrics.json").is_file()
    assert (root / "diagnostics" / "V541_E1_S17" / "metrics.json").is_file()
    assert json.loads(paths["pin"].read_text(encoding="utf-8"))["sha256"] \
        == old_pin_sha


@pytest.mark.parametrize("mode", ["formal", "diagnostic"])
def test_force_refreeze_blocked_by_formal_or_diagnostic_records(
        tmp_path: Path, mode: str) -> None:
    """A mode-labelled formal/diagnostic ATTEMPT refuses, evidence intact."""
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    _make_plan(root, tables, split)
    paths = plan_paths(root)
    old_pin_sha = json.loads(paths["pin"].read_text(encoding="utf-8"))["sha256"]
    records = root / "run_records.jsonl"
    records.write_text(
        json.dumps({"event": "attempt_start", "run_id": "V541_E1_S17",
                    "mode": mode}) + "\n", encoding="utf-8")
    before = records.read_bytes()

    with pytest.raises(Exception) as exc:
        write_plan(root, load_config(), config_path=str(CONFIG),
                   tables_dir=tables, split_path=split, force=True)
    assert getattr(exc.value, "code", "") == "V541_PLAN_STARTED_RUNS"
    assert "V541_E1_S17" in getattr(exc.value, "detail", "")
    assert records.read_bytes() == before
    assert json.loads(paths["pin"].read_text(encoding="utf-8"))["sha256"] \
        == old_pin_sha


@pytest.mark.parametrize("raw", [
    '{"event": "attempt_start", "run_id": "V541_E0_S17"}\n',   # no mode
    '{"event": "attempt_start", "run_id": "V541_E0_S17", "mode": "legacy"}\n',
    '{"event": "attempt_end", "run_id": "V541_E0_S17", "mode": null}\n',
    "not json at all\n",                                        # unreadable
])
def test_unlabelled_or_unreadable_records_block_conservatively(
        tmp_path: Path, raw: str) -> None:
    """Unknown / pre-mode / corrupt ledger lines are never read as 'free'."""
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    _make_plan(root, tables, split)
    records = root / "run_records.jsonl"
    records.write_text(raw, encoding="utf-8")

    classified = classify_started_runs(root)
    assert classified["technical"] == {}
    assert classified["blocking"], "must block conservatively"
    assert all(entry["mode"] in ("unknown", "legacy")
               for entry in classified["blocking"].values())
    with pytest.raises(Exception) as exc:
        write_plan(root, load_config(), config_path=str(CONFIG),
                   tables_dir=tables, split_path=split, force=True)
    assert getattr(exc.value, "code", "") == "V541_PLAN_STARTED_RUNS"


def test_non_attempt_records_do_not_block_a_refreeze(tmp_path: Path) -> None:
    """A REFUSED formal launch is not a started run (no attempt, no dir)."""
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    _make_plan(root, tables, split)
    (root / "run_records.jsonl").write_text(
        json.dumps({"event": "refused", "mode": "formal",
                    "run_id": "V541_E1_S17",
                    "code": "V541_FORMAL_GATES_BLOCKED"}) + "\n",
        encoding="utf-8")
    assert classify_started_runs(root) == {"blocking": {}, "technical": {}}
    result = write_plan(root, load_config(), config_path=str(CONFIG),
                        tables_dir=tables, split_path=split, force=True)
    assert result["rebuilt"] is True
    assert result["refreeze"]["preserved_technical_runs"] == []


def test_plan_pin_tamper_is_refused(tmp_path: Path) -> None:
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    _make_plan(root, tables, split)
    paths = plan_paths(root)
    verify_plan_pin_v541(paths["plan"], paths["pin"])      # intact
    with paths["plan"].open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(Exception) as exc:
        verify_plan_pin_v541(paths["plan"], paths["pin"])
    assert getattr(exc.value, "code", "") == "V541_PLAN_PIN_MISMATCH"


def test_plan_pin_field_drift_is_refused(tmp_path: Path) -> None:
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    _make_plan(root, tables, split)
    paths = plan_paths(root)
    pin = json.loads(paths["pin"].read_text(encoding="utf-8"))
    pin["code_sha256"] = "f" * 64
    paths["pin"].write_text(json.dumps(pin), encoding="utf-8")
    with pytest.raises(Exception) as exc:
        verify_plan_pin_v541(paths["plan"], paths["pin"])
    assert getattr(exc.value, "code", "") in {
        "V541_PLAN_PIN_MISMATCH", "V541_PLAN_PIN_INCOMPLETE"}


# ---------------------------------------------------------------------------
# source fingerprint (code freeze)
# ---------------------------------------------------------------------------
def test_source_fingerprint_changes_with_source(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    target = src / "mod.py"
    target.write_text("X = 1\n", encoding="utf-8")
    contract = {"method": "source_tree_fingerprint",
                "includes": ["src"], "suffixes": [".py"], "excludes": []}
    first = source_manifest(contract, root=tmp_path)
    second = source_manifest(contract, root=tmp_path)
    assert first["code_sha256"] == second["code_sha256"]
    assert first["n_files"] == 1

    target.write_text("X = 2\n", encoding="utf-8")          # source drift
    third = source_manifest(contract, root=tmp_path)
    assert third["code_sha256"] != first["code_sha256"]
    assert third["source_manifest_sha256"] != first["source_manifest_sha256"]

    target.unlink()                                          # file removed
    with pytest.raises(Exception) as exc:
        source_manifest(contract, root=tmp_path)
    assert getattr(exc.value, "code", "") in {"V541_SOURCE_MISSING",
                                              "V541_SOURCE_EMPTY"}
    # a whole missing include entry is its own typed refusal
    with pytest.raises(Exception) as exc:
        source_manifest({"method": "source_tree_fingerprint",
                         "includes": ["ghost"], "suffixes": [".py"],
                         "excludes": []}, root=tmp_path)
    assert getattr(exc.value, "code", "") == "V541_SOURCE_MISSING"


def test_source_fingerprint_excludes_cache_and_bytecode(tmp_path: Path) -> None:
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "a.py").write_text("A = 1\n", encoding="utf-8")
    (tmp_path / "pkg" / "__pycache__").mkdir()
    (tmp_path / "pkg" / "__pycache__" / "b.py").write_text("B = 1",
                                                           encoding="utf-8")
    (tmp_path / "pkg" / "c.txt").write_text("hello\n", encoding="utf-8")
    contract = {"method": "source_tree_fingerprint",
                "includes": ["pkg"], "suffixes": [".py"],
                "excludes": ["__pycache__"]}
    doc = source_manifest(contract, root=tmp_path)
    assert set(doc["files"]) == {"pkg/a.py"}
    assert doc["n_files"] == 1
    assert doc["method"] == "source_tree_fingerprint"
    # the two digests cover different domains (files map vs whole document)
    assert doc["code_sha256"] != doc["source_manifest_sha256"]


# ---------------------------------------------------------------------------
# run: typed refusals
# ---------------------------------------------------------------------------
def test_run_refuses_unknown_embedding(plan_root, capsys) -> None:
    root, tables, _split = plan_root
    code = _cli(root, "run", "--mode", "smoke", "--run-id", "V541_EX_S17",
                "--embedding", "EX", "--seed", "17",
                "--tables-dir", str(tables), "--device", "cpu")
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["code"] == "V541_EMBEDDING_UNKNOWN"


def test_run_refuses_unregistered_candidate(plan_root, capsys) -> None:
    """A preregistered id that cannot be built yet must REFUSE."""
    from pes2ts_core.pesn.models.v541.embedding_registry import (
        FROZEN_EMBEDDING_IDS,
        is_embedding_implemented,
    )

    pending = []
    for eid in FROZEN_EMBEDDING_IDS:
        if eid == "E0":
            continue
        try:
            buildable = is_embedding_implemented(eid)
        except Exception:        # a broken in-flight module is also unrunnable
            buildable = False
        if not buildable:
            pending.append(eid)
    if not pending:
        pytest.skip("every candidate builds — refusal path covered by the "
                    "registry's own tests")
    embedding_id = pending[0]
    root, tables, _split = plan_root
    code = _cli(root, "run", "--mode", "smoke",
                "--run-id", "V541_{}_S17".format(embedding_id),
                "--embedding", embedding_id, "--seed", "17",
                "--tables-dir", str(tables), "--device", "cpu")
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["code"] in {"V541_EMBEDDING_NOT_IMPLEMENTED",
                               "V541_EMBEDDING_IMPORT_FAILED"}
    assert not (root / "smoke").exists()


def test_run_refuses_unregistered_seed(plan_root, capsys) -> None:
    root, tables, _split = plan_root
    code = _cli(root, "run", "--mode", "smoke", "--run-id", "V541_E0_S7",
                "--embedding", "E0", "--seed", "7",
                "--tables-dir", str(tables), "--device", "cpu")
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["code"] == "V541_SEED"


def test_run_refuses_smoke_dose_over_200(plan_root, capsys) -> None:
    root, tables, _split = plan_root
    code = _cli(root, "run", "--mode", "smoke", "--run-id", "V541_E0_S17",
                "--embedding", "E0", "--seed", "17",
                "--tables-dir", str(tables), "--device", "cpu",
                "--stop-after", "500")
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["code"] == "V541_SMOKE_DOSE"


# ---------------------------------------------------------------------------
# technical run_id -> unique matching plan line (full SHA block)
# ---------------------------------------------------------------------------
def _plan_for(root: Path, tables: Path, split: Path) -> dict:
    from scripts.campaigns.v541_runplan import read_plan as _read

    return _read(plan_paths(root)["plan"])


def _ns(**over) -> "argparse.Namespace":
    import argparse

    base = dict(run_id="V541_E0_S17", embedding="E0", seed=17, mode="smoke")
    base.update(over)
    return argparse.Namespace(**base)


def test_technical_run_id_binds_the_t0r_line_shas(tmp_path: Path) -> None:
    """E0 smoke id ``V541_E0_S17`` has no exact plan row — it must still
    bind the FULL provenance block of the unique T0-R line."""
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    _make_plan(root, tables, split)
    plan = _plan_for(root, tables, split)

    identity = R.resolve_run_identity(_ns(), load_config(), plan)
    assert identity["in_plan"] is False
    assert identity["bound_plan_run_id"] == "V541_T0R_E0_S17"
    line = identity["line"]
    for field in ("common_protocol_sha256", "config_sha256", "code_sha256",
                  "environment_sha256", "data_manifest_sha256",
                  "split_sha256", "label_spec_sha256", "r3_init_sha256",
                  "budget_sha256"):
        assert isinstance(line.get(field), str) and len(line[field]) == 64, field
    assert line["embedding_id"] == "E0" and line["seed"] == 17


def test_candidate_technical_run_id_binds_its_exact_line(tmp_path: Path) -> None:
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    _make_plan(root, tables, split)
    plan = _plan_for(root, tables, split)

    identity = R.resolve_run_identity(
        _ns(run_id="V541_E1_S17", embedding="E1"), load_config(), plan)
    assert identity["in_plan"] is True
    assert identity["bound_plan_run_id"] == "V541_E1_S17"
    assert identity["line"]["common_protocol_sha256"]


def test_t4_diagnostic_rows_never_steal_the_binding(tmp_path: Path) -> None:
    """T4 shares E0+seed17; the binding must pick the T0-R line, uniquely."""
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    _make_plan(root, tables, split)
    plan = _plan_for(root, tables, split)

    bound = R.plan_line_for(plan, "E0", 17)
    assert bound["run_id"] == "V541_T0R_E0_S17"
    with pytest.raises(Exception) as exc:
        R.plan_line_for(plan, "E0", 99)
    assert exc.value.code == "V541_RUN_IDENTITY_UNREGISTERED"


def test_formal_still_requires_an_exact_plan_line(tmp_path: Path) -> None:
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    _make_plan(root, tables, split)
    plan = _plan_for(root, tables, split)

    with pytest.raises(Exception) as exc:
        R.resolve_run_identity(_ns(mode="formal"), load_config(), plan)
    assert exc.value.code == "V541_RUN_NOT_IN_PLAN"


# ---------------------------------------------------------------------------
# R3 re-evaluation gate (baseline verify report) + WP2 parity as its own gate
# ---------------------------------------------------------------------------
def test_r3_gate_uses_the_baseline_verify_report(tmp_path: Path) -> None:
    root = _root(tmp_path)
    audits = root / "audits"
    audits.mkdir(parents=True, exist_ok=True)
    (audits / "verify_report.json").write_text(
        json.dumps({"schema": "v541_baseline_verify_v1", "pass": True}),
        encoding="utf-8")

    report = R.t1_gate_report(load_config(), root=root, embedding_id="E0")
    codes = {b["code"] for b in report["blocks"]}
    assert "V541_GATE_R3_REEVAL" not in codes, report["blocks"]
    # the WP2 E0 parity file is a SEPARATE gate and is still missing here
    assert "V541_GATE_WP2_PARITY" in codes

    (audits / "verify_report.json").write_text(
        json.dumps({"schema": "v541_baseline_verify_v1", "pass": False}),
        encoding="utf-8")
    report = R.t1_gate_report(load_config(), root=root, embedding_id="E0")
    assert "V541_GATE_R3_REEVAL" in {b["code"] for b in report["blocks"]}


def test_r3_gate_fails_closed_without_the_verify_report(tmp_path: Path) -> None:
    root = _root(tmp_path)
    # a passing WP2 parity file alone must NOT satisfy the R3 re-eval gate
    audits = root / "audits"
    audits.mkdir(parents=True, exist_ok=True)
    (audits / "v53_parity.json").write_text(
        json.dumps({"passed": True}), encoding="utf-8")
    report = R.t1_gate_report(load_config(), root=root, embedding_id="E0")
    codes = {b["code"] for b in report["blocks"]}
    assert "V541_GATE_R3_REEVAL" in codes
    assert "V541_GATE_WP2_PARITY" not in codes


def test_run_refuses_non_primary_smoke_seed(plan_root, capsys) -> None:
    root, tables, _split = plan_root
    code = _cli(root, "run", "--mode", "smoke", "--run-id", "V541_E0_S29",
                "--embedding", "E0", "--seed", "29",
                "--tables-dir", str(tables), "--device", "cpu")
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["code"] == "V541_SMOKE_SEED"


def test_run_refuses_stop_after_on_formal(plan_root, capsys) -> None:
    root, tables, _split = plan_root
    code = _cli(root, "run", "--mode", "formal", "--run-id", "V541_E1_S17",
                "--embedding", "E1", "--seed", "17",
                "--tables-dir", str(tables), "--device", "cpu",
                "--stop-after", "100")
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    # E1 is not registered yet: the registry refusal fires first (typed)
    assert payload["code"] in {"V541_EMBEDDING_NOT_IMPLEMENTED",
                               "V541_FORMAL_DOSE"}


def test_run_requires_a_pinned_plan(tmp_path: Path, capsys) -> None:
    root = _root(tmp_path)
    tables, _split = _stub_tables(tmp_path)
    code = _cli(root, "run", "--mode", "smoke", "--run-id", "V541_E0_S17",
                "--embedding", "E0", "--seed", "17",
                "--tables-dir", str(tables), "--device", "cpu")
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["code"] == "V541_PLAN_MISSING"


# ---------------------------------------------------------------------------
# run: formal is fail-closed
# ---------------------------------------------------------------------------
def test_formal_run_lists_typed_blocks_and_writes_nothing(
        plan_root, capsys) -> None:
    root, tables, _split = plan_root
    code = _cli(root, "run", "--mode", "formal",
                "--run-id", "V541_T0R_E0_S17", "--embedding", "E0",
                "--seed", "17", "--tables-dir", str(tables),
                "--device", "cpu")
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["code"] == "V541_FORMAL_GATES_BLOCKED"
    codes = {b["code"] for b in payload["blocks"]}
    assert "V541_GATE_WP1" in codes
    assert "V541_GATE_R3_REEVAL" in codes
    assert "V541_GATE_E0_DECISION" in codes
    assert "V541_GATE_H0" in codes
    assert "V541_GATE_H1" in codes
    assert "V541_GATE_SMOKE_200" in codes
    # the frozen config AND the instantiated stub budget are signed now, so
    # neither the config-level blocker nor the budget-signoff gate fires
    # (see test_formal_run_with_null_budget_config_is_blocked for the
    # explicit null-budget path)
    assert "V541_T1_CONFIG" not in codes
    assert "V541_GATE_BUDGET_SIGNOFF" not in codes
    # no formal directory, no attempt
    assert not (root / "runs").exists()
    events = [r["event"] for r in _records(root)]
    assert events == ["refused"]


def test_formal_run_with_null_budget_config_is_blocked(tmp_path: Path,
                                                       capsys) -> None:
    """A NULL per-candidate budget still refuses a FORMAL launch.

    Built on an explicit temporary config copy — the frozen repo config is
    signed, and its bytes (and the real ``outputs/`` tree) are never
    touched by this fixture.
    """
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    null_cfg = _null_budget_config(tmp_path)
    from scripts.agent_ops.v541_config_schema import load_config as _load

    _make_plan(root, tables, split, config_path=null_cfg,
               cfg=_load(str(null_cfg)))
    code = _cli(root, "run", "--mode", "formal",
                "--run-id", "V541_T0R_E0_S17", "--embedding", "E0",
                "--seed", "17", "--config", str(null_cfg),
                "--tables-dir", str(tables), "--device", "cpu")
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["code"] == "V541_FORMAL_GATES_BLOCKED"
    config_blocks = [b for b in payload["blocks"]
                     if b["code"] == "V541_T1_CONFIG"]
    assert config_blocks, payload["blocks"]
    assert "per_candidate_core_time" in config_blocks[0]["detail"]
    assert not (root / "runs").exists()


def test_formal_refuses_when_source_drifts_after_pin(tmp_path: Path,
                                                     monkeypatch,
                                                     capsys) -> None:
    root = _root(tmp_path)
    tables, split = _stub_tables(tmp_path)
    _make_plan(root, tables, split)
    pinned = json.loads(plan_paths(root)["pin"].read_text(encoding="utf-8"))
    assert pinned["code_sha256"] != "e" * 64
    # simulate a source edit AFTER the plan freeze (code fingerprint drift)
    monkeypatch.setattr(R, "code_sha256", lambda *a, **k: "e" * 64)
    code = _cli(root, "run", "--mode", "smoke", "--run-id", "V541_E0_S17",
                "--embedding", "E0", "--seed", "17",
                "--tables-dir", str(tables), "--device", "cpu")
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["code"] == "V541_CODE_DRIFT"
    assert not (root / "smoke").exists()


# ---------------------------------------------------------------------------
# run: smoke / profile / records (fixture data)
# ---------------------------------------------------------------------------
def test_smoke_leg_then_resume_appends_attempts(tmp_path: Path,
                                                plan_root,
                                                fixture_inputs,
                                                capsys) -> None:
    root, tables, _split = plan_root
    lock = tmp_path / "gpu.lock"
    common = _run_args(tables, lock)

    code = _cli(root, *common, "--mode", "smoke",
                "--stop-after", "2")
    first = json.loads(capsys.readouterr().out)
    assert code == 0, first
    assert first["mode"] == "smoke"
    assert first["is_formal_run"] is False
    assert first["run_outcome"] == "done"
    assert first["attempt_status"] == "leg_partial"
    assert first["target_updates"] == 200
    assert first["leg_window_updates"] == 2

    code = _cli(root, *common, "--mode", "smoke",
                "--stop-after", "4", "--resume")
    second = json.loads(capsys.readouterr().out)
    assert code == 0, second
    assert second["attempt"] == 2
    assert second["updates_run"] == 2          # resumed 3 -> 4, no restart

    out_dir = root / "smoke" / "V541_E0_S17"
    assert out_dir.is_dir()
    assert (out_dir / "metrics.json").is_file()
    assert (out_dir / "manifest.jsonl").is_file()

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["is_formal_run"] is False
    assert metrics["dev_eval_done"] is False
    assert metrics["confirm_read"] is False
    assert metrics["dev_positions_used_for"] == "curve_exclusion_only"
    assert metrics["evaluation_scope"] == "inner_val"
    assert metrics["schedule"]["total_updates"] == 4

    records = _records(root)
    assert [r["event"] for r in records] == [
        "attempt_start", "attempt_end", "attempt_start", "attempt_end"]
    assert records[0]["new_qc_calls"] == 0 and records[0]["confirm_read"] is False
    # the technical smoke id has no exact plan row — it must still carry the
    # FULL provenance block of the bound T0-R line (never empty SHAs)
    assert records[0]["in_plan"] is False
    assert records[0]["bound_plan_run_id"] == "V541_T0R_E0_S17"
    for field in ("common_protocol_sha256", "code_sha256",
                  "environment_sha256", "config_sha256"):
        assert isinstance(records[0].get(field), str) \
            and len(records[0][field]) == 64, (field, records[0].get(field))
    assert records[1]["checkpoint_best_sha256"]
    assert len(records[1]["batch_schedule_sha256"]) == 64
    assert records[1]["tau"]["calibrated_on"] == "inner_val"
    assert records[1]["cost"]["wall_clock_s"] is not None
    assert records[3]["attempt"] == 2

    # the legacy V5.3-MP ledger is never opened for writing
    legacy = tmp_path / "outputs" / "pesn" / "v53mp" / "run_records.jsonl"
    assert not legacy.exists()


def test_smoke_resume_uses_matching_identity(tmp_path: Path,
                                             plan_root,
                                             fixture_inputs,
                                             capsys) -> None:
    root, tables, _split = plan_root
    lock = tmp_path / "gpu.lock"
    common = _run_args(tables, lock)
    _cli(root, *common, "--mode", "smoke", "--stop-after", "2")
    capsys.readouterr()
    _cli(root, *common, "--mode", "smoke", "--stop-after", "4")
    capsys.readouterr()

    import torch

    payload = torch.load(str(root / "smoke" / "V541_E0_S17"
                             / "checkpoint_last.pt"),
                         map_location="cpu", weights_only=False)
    identity = payload["identity"]
    assert identity["degree_stats_sha256"] is None
    assert identity["config_sha256"]
    assert identity["sampler_spec_sha256"]
    assert payload["update"] == 4


def test_smoke_dose_guard_and_fresh_checkpoint_guard(tmp_path: Path,
                                                     plan_root,
                                                     fixture_inputs,
                                                     capsys) -> None:
    root, tables, _split = plan_root
    lock = tmp_path / "gpu.lock"
    common = _run_args(tables, lock)
    _cli(root, *common, "--mode", "smoke", "--stop-after", "2")
    capsys.readouterr()
    # restarting from scratch over a live checkpoint is refused
    code = _cli(root, *common, "--mode", "smoke", "--fresh",
                "--stop-after", "2")
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["code"] == "V541_CHECKPOINT_EXISTS"


def test_profile_window_is_never_formal(tmp_path: Path, plan_root,
                                        fixture_inputs, capsys) -> None:
    root, tables, _split = plan_root
    lock = tmp_path / "gpu.lock"
    common = _run_args(tables, lock)
    code = _cli(root, *common, "--mode", "profile",
                "--profile-w0", "2", "--profile-w1", "2")
    payload = json.loads(capsys.readouterr().out)
    assert code == 0, payload
    assert payload["mode"] == "profile"
    assert payload["is_formal_run"] is False

    out_dir = root / "profile" / "V541_E0_S17"
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["profile_only"] is True
    assert metrics["is_formal_run"] is False
    assert metrics["run_outcome"] is None            # profile has no outcome
    assert metrics["profile_window"] == {"profile_w0_updates": 2,
                                         "profile_w1_updates": 2,
                                         "total_updates": 4}
    assert metrics["formal_schedule_untouched"]["total_updates"] == 37900

    record = [r for r in _records(root) if r["event"] == "attempt_end"][0]
    assert record["is_formal_run"] is False


def test_profile_w0_window_may_not_exceed_400(plan_root, capsys) -> None:
    root, tables, _split = plan_root
    code = _cli(root, "run", "--mode", "profile", "--run-id", "V541_E0_S17",
                "--embedding", "E0", "--seed", "17",
                "--tables-dir", str(tables), "--device", "cpu",
                "--profile-w0", "401", "--profile-w1", "1")
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["code"] == "V541_PROFILE_WINDOW"


def test_run_refuses_after_config_drift(plan_root, capsys) -> None:
    root, tables, _split = plan_root
    text = CONFIG.read_text(encoding="utf-8")
    try:
        CONFIG.write_text(text.replace("version: 0.1.0-wp0",
                                       "version: 0.1.0-wp0-drifted"),
                          encoding="utf-8")
        code = _cli(root, "run", "--mode", "smoke", "--run-id", "V541_E0_S17",
                    "--embedding", "E0", "--seed", "17",
                    "--tables-dir", str(tables), "--device", "cpu")
        payload = json.loads(capsys.readouterr().out)
        assert code == 1
        # the campaign yaml is part of the source fingerprint, so a config
        # edit surfaces as code drift (and would ALSO be CONFIG_DRIFT)
        assert payload["code"] in {"V541_CODE_DRIFT", "V541_CONFIG_DRIFT"}
    finally:
        CONFIG.write_text(text, encoding="utf-8")


def test_run_refuses_after_data_manifest_drift(plan_root, capsys) -> None:
    root, tables, _split = plan_root
    manifest = tables / "data_manifest.json"
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    doc["files"][0]["sha256"] = "0" * 64
    manifest.write_text(json.dumps(doc, sort_keys=True), encoding="utf-8")
    code = _cli(root, "run", "--mode", "smoke", "--run-id", "V541_E0_S17",
                "--embedding", "E0", "--seed", "17",
                "--tables-dir", str(tables), "--device", "cpu")
    payload = json.loads(capsys.readouterr().out)
    assert code == 1
    assert payload["code"] == "V541_DATA_DRIFT"


# ---------------------------------------------------------------------------
# gate-check
# ---------------------------------------------------------------------------
def test_gate_check_fails_closed_without_evidence(plan_root, capsys) -> None:
    root, tables, _split = plan_root
    code = _cli(root, "gate-check", "--tables-dir", str(tables))
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["ok"] is False
    assert "idx_v2_binding" in report["failed"]        # stub has no idx_v2
    assert any(name.startswith("t1_gate:V541_GATE_")
               for name in report["failed"])
    assert report["items"]["new_qc_calls"] == "pass"
    assert report["items"]["runplan_pin"] == "pass"


def test_gate_check_reports_pin_tamper(plan_root, capsys) -> None:
    root, tables, _split = plan_root
    with plan_paths(root)["plan"].open("a", encoding="utf-8") as handle:
        handle.write("\n")
    code = _cli(root, "gate-check", "--tables-dir", str(tables))
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["items"]["runplan_pin"] == "fail"
    assert "PLAN_SHA_MISMATCH" in report["details"]["runplan_pin"]


# ---------------------------------------------------------------------------
# old-ledger isolation
# ---------------------------------------------------------------------------
def test_legacy_v53mp_ledger_is_never_written(tmp_path: Path,
                                              plan_root,
                                              fixture_inputs,
                                              capsys) -> None:
    legacy = tmp_path / "outputs" / "pesn" / "v53mp" / "run_records.jsonl"
    legacy.parent.mkdir(parents=True, exist_ok=True)
    sentinel = '{"campaign_id": "PES2TS_V53MP_MOVE_PROCESS", "n": 1}\n'
    legacy.write_text(sentinel, encoding="utf-8")

    root, tables, _split = plan_root
    lock = tmp_path / "gpu.lock"
    common = _run_args(tables, lock)
    _cli(root, *common, "--mode", "smoke", "--stop-after", "2")
    capsys.readouterr()

    assert legacy.read_text(encoding="utf-8") == sentinel
    # ... while V541's own ledger got the attempt
    records = _records(root)
    assert records and records[0]["run_id"] == "V541_E0_S17"
    assert root.name == "v541_embed_50k"
