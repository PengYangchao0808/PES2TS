"""Infra bench acceptance (plan §7, scripts/infra/infra_bench.py).

Covers the pure parse/verdict/report logic on synthetic train logs, the
bench-root mechanism (plan row derivation + driver-style plan/pin writing,
validated with the driver's own verify_plan_pin), the r3-CSV sampler
parsing, and the ``--dry-run`` end-to-end path.  No GPU, no driver run, no
ledger writes into the campaign root.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.agent_ops.ledger.run_ledger import (  # noqa: E402
    read_run_plan,
    verify_plan_pin,
)
from scripts.infra import infra_bench as ib  # noqa: E402


def _synthetic_log(updates: int = 1000, step_s: float = 0.078,
                   warmup_step_s: float = 0.150,
                   val_every: int = 100) -> list[str]:
    """train_log.jsonl-like rows: update 1, every val_every, plus the
    non-timed grad_monitor / inner_val_loss rows the real trainer emits."""
    lines = []
    elapsed = 1.2  # startup-ish first update
    lines.append(json.dumps({"update": 1, "phase": "joint", "loss": 0.25,
                             "elapsed_s": round(elapsed, 3)}))
    for u in range(val_every, updates + 1, val_every):
        elapsed += val_every * (warmup_step_s if u <= 200 else step_s)
        lines.append(json.dumps({"update": u, "phase": "joint", "loss": 0.24,
                                 "elapsed_s": round(elapsed, 3)}))
        lines.append(json.dumps({"update": u,
                                 "grad_monitor": {"encoder_share": 0.0}}))
        lines.append(json.dumps({"update": u, "inner_val_loss": 0.23}))
    return lines


# ---------------------------------------------------------------------------
# train_log parsing
# ---------------------------------------------------------------------------


def test_parse_train_log_excludes_warmup() -> None:
    lines = _synthetic_log(updates=1000, step_s=0.078, warmup_step_s=0.150)
    stats = ib.parse_train_log(lines, warmup=200)
    assert stats["n_updates"] == 1000
    # timed rows only (grad_monitor/inner_val_loss rows carry no elapsed_s)
    assert stats["n_timed"] == 11
    assert stats["ms_update"], "post-warmup intervals expected"
    # every counted interval is the 0.078 s one; the 0.150 warmup ones are out
    assert all(abs(v - 78.0) < 1e-6 for v in stats["ms_update"])
    assert abs(stats["median_ms"] - 78.0) < 1e-6
    assert abs(stats["p95_ms"] - 78.0) < 1e-6
    # warmup exclusion applies to ms/update ONLY — the first row (update 1)
    # always reports its elapsed_s regardless of warmup
    assert stats["first_row_elapsed_s"] == 1.2


def test_parse_train_log_warmup_included_when_low() -> None:
    lines = _synthetic_log(updates=400, step_s=0.078, warmup_step_s=0.150)
    cold = ib.parse_train_log(lines, warmup=0)
    warm = ib.parse_train_log(lines, warmup=200)
    assert cold["median_ms"] > warm["median_ms"], \
        "warmup=0 must drag the median up with the slow first intervals"
    assert warm["median_ms"] == pytest.approx(78.0, abs=1e-6)


def test_parse_train_log_ignores_garbage() -> None:
    lines = ["not json", "", '{"no_update": true}'] + _synthetic_log(300)
    stats = ib.parse_train_log(lines, warmup=200)
    assert stats["n_updates"] == 300


def test_parse_train_log_matches_real_log(tmp_path: Path) -> None:
    """Parse a REAL S3 train_log and confirm the timing field contract."""
    real = REPO / "outputs/pesn/v53mp/v53mp_20k_A_s17/train_log.jsonl"
    if not real.is_file():
        pytest.skip("real S3 train_log not present")
    lines = real.read_text(encoding="utf-8").splitlines()
    stats = ib.parse_train_log(lines, warmup=200)
    assert stats["n_updates"] == 15400
    assert stats["n_timed"] >= 150          # update 1 + every 100 + final
    assert stats["first_row_elapsed_s"] is not None
    assert stats["median_ms"] is not None and stats["median_ms"] > 0
    assert stats["p95_ms"] >= stats["median_ms"]


# ---------------------------------------------------------------------------
# r3 CSV sampler parsing (cgroup_scope.sh / r3_v53mp_cgroup_run.sh format)
# ---------------------------------------------------------------------------


def test_parse_cgroup_samples_r3_csv() -> None:
    csv_lines = [
        "ts,memory_current_bytes,anon_bytes,file_bytes",          # 4-col r3
        "12:00:00,10,5,3",
        "12:00:02,30,25,3",
        "12:00:04,20,15,3",
    ]
    stats = ib.parse_cgroup_samples(csv_lines, events_text="oom 1\noom_kill 1\n")
    assert stats == {"anon_peak_bytes": 25, "swap_max_bytes": 0,
                     "oom_kill": 1}


def test_parse_cgroup_samples_scope_csv_with_swap() -> None:
    csv_lines = [
        "ts,memory_current_bytes,anon_bytes,file_bytes,memory_swap_current_bytes",
        "12:00:00,10,5,3,0",
        "12:00:02,30,25,3,4096",
        "12:00:04,20,15,3,0",
    ]
    stats = ib.parse_cgroup_samples(csv_lines, events_text="oom_kill 0\n")
    assert stats["anon_peak_bytes"] == 25
    assert stats["swap_max_bytes"] == 4096
    assert stats["oom_kill"] == 0


def test_parse_cgroup_samples_tolerates_garbage() -> None:
    stats = ib.parse_cgroup_samples(["", "junk", "ts,a,b,c"], "")
    assert stats == {"anon_peak_bytes": 0, "swap_max_bytes": 0,
                     "oom_kill": 0}


def test_parse_vram_samples() -> None:
    stats = ib.parse_vram_samples(["2617 MiB", "3050 MiB", "", "garbage"])
    assert stats["vram_peak_mib"] == 3050
    assert stats["vram_samples"] == 2


# ---------------------------------------------------------------------------
# verdict + report
# ---------------------------------------------------------------------------


def test_verdict_b0_pass_and_fail() -> None:
    ok = ib.verdict_b0(median_ms=78.0, anon_peak_bytes=int(26.0e9),
                       swap_max_bytes=0, oom_kill=0)
    assert ok["pass"] is True
    assert ok["gate"] == "B0"
    assert set(ok["checks"]) == {"median_ms_update_in_band",
                                 "anon_peak_in_band", "swap_gate", "oom_gate"}

    slow = ib.verdict_b0(median_ms=120.0, anon_peak_bytes=int(26.0e9),
                         swap_max_bytes=0, oom_kill=0)
    assert slow["pass"] is False
    assert slow["checks"]["median_ms_update_in_band"] is False

    leaky = ib.verdict_b0(median_ms=78.0, anon_peak_bytes=int(40.0e9),
                          swap_max_bytes=0, oom_kill=0)
    assert leaky["pass"] is False
    assert leaky["checks"]["anon_peak_in_band"] is False

    swapped = ib.verdict_b0(median_ms=78.0, anon_peak_bytes=int(26.0e9),
                            swap_max_bytes=4096, oom_kill=0)
    assert swapped["pass"] is False
    assert swapped["checks"]["swap_gate"] is False

    killed = ib.verdict_b0(median_ms=78.0, anon_peak_bytes=int(26.0e9),
                           swap_max_bytes=0, oom_kill=3)
    assert killed["pass"] is False
    assert killed["checks"]["oom_gate"] is False


def test_render_report_md_smoke() -> None:
    report = ib.build_report(
        tag="B0", run_plan="plan.jsonl", run_id="v53mp_20k_A_s17",
        bench_root="/x/root", gate_check="pass", data_root="data",
        requested_updates=2000, driver_effective_updates=2000,
        warmup=200, dry_run=True,
        rep_reports=[{
            "rep": 0, "median_ms": 78.0, "p95_ms": 82.0,
            "startup_s": 12.0, "anon_peak_bytes": int(26.0e9),
            "swap_max_bytes": 0, "oom_kill": 0, "vram_peak_mib": 2500,
            "exit": 0, "gate_exit": 0}],
        warnings=[],
        ms_target=78.0, ms_tol=0.15, anon_target_gb=26.0, anon_tol=0.15,
        config_source="bench_derived",
        original_config_sha256="a" * 64,
        bench_config_sha256="b" * 64)
    md = ib.render_report_md(report)
    assert "# Infra bench B0" in md
    assert "median ms/update" in md
    assert "gate_check: **pass**" in md
    assert "driver-effective updates: 2000" in md
    assert "config: **bench_derived**" in md
    assert "aaaa" in md and "bbbb" in md
    assert "## Verdict" in md
    assert report["verdict"]["pass"] is True
    assert report["aggregate"]["median_ms_update"] == 78.0
    assert report["config_source"] == "bench_derived"
    assert report["original_config_sha256"] == "a" * 64
    assert report["bench_config_sha256"] == "b" * 64


def test_render_report_md_original_config_line() -> None:
    report = ib.build_report(
        tag="B0", run_plan="p", run_id="v53mp_20k_A_s17",
        bench_root="/x", gate_check="pass", data_root="d",
        requested_updates=15400, driver_effective_updates=15400,
        warmup=200, dry_run=True, rep_reports=[], warnings=[],
        ms_target=78.0, ms_tol=0.15, anon_target_gb=26.0, anon_tol=0.15,
        config_source="original",
        original_config_sha256="c" * 64,
        bench_config_sha256=None)
    md = ib.render_report_md(report)
    assert "config: **original** (sha cccccccccccc)" in md
    assert report["config_source"] == "original"
    assert report["bench_config_sha256"] is None


# ---------------------------------------------------------------------------
# bench-root mechanism (plan row + plan/pin per driver scheme)
# ---------------------------------------------------------------------------


def _source_row() -> dict:
    return {
        "run_id": "v53mp_20k_A_s17", "campaign_id": "PES2TS_V53MP_MOVE_PROCESS",
        "phase": "S3", "tier": "20k", "arm": "A", "fold": 0, "seed": 17,
        "n_updates": 15400, "smoke_updates": None, "status": "planned",
        "confirm": False, "r3_init": "5372d9b12807c3f0",
        "config_sha256": "0" * 64, "data_sha256": "1" * 64,
        "teacher_sha256": "2" * 64, "features_sha256": "3" * 64,
        "split_sha256": "4" * 64,
    }


def test_build_bench_row_overrides_only_intent_fields() -> None:
    source = _source_row()
    row = ib.build_bench_row(source, tag="B0", updates=2000)
    assert row["n_updates"] == 2000
    assert row["smoke_updates"] is None
    assert row["status"] == "planned"
    assert row["infra_bench"]["tag"] == "B0"
    # identity/shas/run_id copied VERBATIM (driver _parse_run_id contract)
    for field in ("run_id", "campaign_id", "phase", "tier", "arm", "seed",
                  "config_sha256", "data_sha256", "teacher_sha256",
                  "features_sha256", "split_sha256"):
        assert row[field] == source[field], field
    # source dict untouched
    assert source["n_updates"] == 15400


def test_build_bench_row_config_sha_override() -> None:
    source = _source_row()
    row = ib.build_bench_row(source, tag="B0", updates=2000,
                             config_sha256="f" * 64)
    assert row["config_sha256"] == "f" * 64   # binds to the bench yaml
    assert source["config_sha256"] == "0" * 64  # source untouched
    # all other shas stay verbatim
    for field in ("data_sha256", "teacher_sha256", "features_sha256",
                  "split_sha256"):
        assert row[field] == source[field], field


# ---------------------------------------------------------------------------
# bench config derivation (closes the 15400-vs-2000 run-length gap)
# ---------------------------------------------------------------------------


def _source_config(tmp_path: Path) -> Path:
    """Minimal campaign yaml mirroring the frozen one's schedule shape."""
    import yaml

    doc = {
        "schema": "pes2ts_v53mp_campaign_v1",
        "campaign_id": "PES2TS_V53MP_MOVE_PROCESS",
        "training_contract": {
            "schedule": {
                "staged": True,
                "freeze_encoder_updates": 400,
                "joint_updates_base_2k": 1500,
                "joint_updates_scaling": "density_preserving_per_tier",
                "lr_head": 3e-4,
                "lr_encoder": 1e-4,
                "weight_decay": 1e-4,
                "grad_clip": 1.0,
                "precision": "fp32",
            },
            "lambda_probe": {"values": {"lambda_rank": 0.5,
                                        "lambda_q": 0.5,
                                        "lambda_endpoint": 0.0123}},
            "sampling": {"batch_curves": 8, "frames_per_curve": 4},
            "inner_val_max_samples": 8192,
        },
    }
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path


def test_derive_bench_config_rewrites_schedule_only(tmp_path: Path) -> None:
    import yaml

    from pes2ts_core.utils.hashing import sha256_file

    original = _source_config(tmp_path)
    original_bytes = original.read_bytes()
    out = tmp_path / "root" / "config_bench.yaml"
    derived = ib.derive_bench_config(original, "20k", 2000, out)

    assert out.is_file()
    assert derived["sha256"] == sha256_file(out)
    assert derived["freeze_encoder_updates"] == 400
    assert derived["joint_updates_base_2k"] == 160   # (2000-400)/10
    assert derived["tier_factor"] == 10
    assert derived["driver_effective_updates"] == 2000

    doc = yaml.safe_load(out.read_text(encoding="utf-8"))
    schedule = doc["training_contract"]["schedule"]
    assert schedule["freeze_encoder_updates"] == 400   # freeze stays
    assert schedule["joint_updates_base_2k"] == 160
    # every other knob verbatim
    assert schedule["lr_head"] == 3e-4
    assert schedule["grad_clip"] == 1.0
    assert doc["training_contract"]["lambda_probe"]["values"] == {
        "lambda_rank": 0.5, "lambda_q": 0.5, "lambda_endpoint": 0.0123}
    assert doc["training_contract"]["sampling"] == {
        "batch_curves": 8, "frames_per_curve": 4}
    assert doc["schema"] == "pes2ts_v53mp_campaign_v1"
    # provenance marker records the derivation inputs
    assert doc["infra_bench"]["derived_from"] == str(original)
    assert doc["infra_bench"]["requested_updates"] == 2000
    # driver-effective from the DERIVED file == --updates (self-check)
    assert ib.driver_effective_updates(out, "20k") == 2000
    # original yaml never written
    assert original.read_bytes() == original_bytes


def test_derive_bench_config_5k_factor(tmp_path: Path) -> None:
    out = tmp_path / "config_bench.yaml"
    # tier 5k factor = 5000//2000 = 2 -> joint 1600 = base 800 * 2
    derived = ib.derive_bench_config(_source_config(tmp_path), "5k", 2000,
                                     out)
    assert derived["joint_updates_base_2k"] == 800
    assert derived["driver_effective_updates"] == 2000


def test_derive_bench_config_refuses_updates_le_freeze(tmp_path: Path) -> None:
    out = tmp_path / "config_bench.yaml"
    with pytest.raises(SystemExit, match="UPDATES_LE_FREEZE"):
        ib.derive_bench_config(_source_config(tmp_path), "20k", 400, out)
    with pytest.raises(SystemExit, match="UPDATES_LE_FREEZE"):
        ib.derive_bench_config(_source_config(tmp_path), "20k", 399, out)
    assert not out.exists()   # nothing written on refusal


def test_derive_bench_config_refuses_indivisible(tmp_path: Path) -> None:
    out = tmp_path / "config_bench.yaml"
    # tier 20k factor 10: (2005-400)=1605 is not divisible by 10
    with pytest.raises(SystemExit, match="UPDATES_NOT_DIVISIBLE"):
        ib.derive_bench_config(_source_config(tmp_path), "20k", 2005, out)
    assert not out.exists()


def test_derive_bench_config_uses_custom_freeze(tmp_path: Path) -> None:
    import yaml

    original = _source_config(tmp_path)
    doc = yaml.safe_load(original.read_text(encoding="utf-8"))
    doc["training_contract"]["schedule"]["freeze_encoder_updates"] = 0
    original.write_text(yaml.safe_dump(doc, sort_keys=False),
                        encoding="utf-8")
    out = tmp_path / "config_bench.yaml"
    derived = ib.derive_bench_config(original, "20k", 2000, out)
    assert derived["freeze_encoder_updates"] == 0
    assert derived["joint_updates_base_2k"] == 200   # (2000-0)/10


def test_validate_bench_root_accepts_derived_config(tmp_path: Path) -> None:
    """The pin/row sha chain must validate against the DERIVED config."""
    from pes2ts_core.utils.hashing import sha256_file

    tables = tmp_path / "data"
    tables.mkdir()
    (tables / "data_manifest.json").write_text('{"files": []}\n',
                                               encoding="utf-8")
    (tables / "teacher_manifest.json").write_text('{"files": []}\n',
                                                  encoding="utf-8")
    split = tables / "split_manifest_v53mp.json"
    split.write_text('{"schema": "v53mp_split_v1"}\n', encoding="utf-8")

    original = _source_config(tmp_path)
    bench_root = tmp_path / "root"
    derived = ib.derive_bench_config(original, "20k", 2000,
                                     bench_root / "config_bench.yaml")
    row = _source_row()
    row["data_sha256"] = sha256_file(tables / "data_manifest.json")
    row["teacher_sha256"] = sha256_file(tables / "teacher_manifest.json")
    row["split_sha256"] = sha256_file(split)
    row["features_sha256"] = sha256_file(
        REPO / "pes2ts_core/pesn/features/key_atom_graph_v3.py")
    row = ib.build_bench_row(row, "B0", 2000,
                             config_sha256=derived["sha256"])
    header = ib.build_bench_header(None, row, "B0")
    plan_path, pin_path = ib.write_bench_plan(bench_root, row, header)

    # pin binds the plan; the plan row's config_sha256 == derived sha
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    verify_plan_pin(plan_path, pin)
    plan_row = read_run_plan(plan_path)["by_run_id"]["v53mp_20k_A_s17"]
    assert plan_row["config_sha256"] == derived["sha256"]

    result = ib.validate_bench_root(
        bench_root, plan_path, "v53mp_20k_A_s17",
        Path(derived["path"]), tables, split)
    assert result["five_sha"] == "ok"

    # sanity: binding the row to the ORIGINAL sha while validating with the
    # DERIVED config must be refused (wrong-yaml binding is a real error)
    bad_source = _source_row()
    bad_source["data_sha256"] = sha256_file(tables / "data_manifest.json")
    bad_source["teacher_sha256"] = sha256_file(
        tables / "teacher_manifest.json")
    bad_source["split_sha256"] = sha256_file(split)
    bad_source["features_sha256"] = sha256_file(
        REPO / "pes2ts_core/pesn/features/key_atom_graph_v3.py")
    bad_row = ib.build_bench_row(bad_source, "B0", 2000,
                                 config_sha256=sha256_file(original))
    bad_header = ib.build_bench_header(None, bad_row, "B0")
    bad_root = tmp_path / "bad_root"
    bad_plan, _ = ib.write_bench_plan(bad_root, bad_row, bad_header)
    with pytest.raises(SystemExit):
        ib.validate_bench_root(bad_root, bad_plan, "v53mp_20k_A_s17",
                               Path(derived["path"]), tables, split)


def test_build_bench_header_keeps_campaign_mode() -> None:
    source_header = {"record_type": "plan_header",
                     "campaign_mode": "full", "tier": "20k", "phase": "S3",
                     "sha_status": {"config_sha256": "real"}}
    row = _source_row()
    header = ib.build_bench_header(source_header, row, "B0")
    assert header["campaign_mode"] == "full"   # pin cross-check depends on it
    assert header["record_type"] == "plan_header"
    assert header["infra_bench"] is True
    # synthesized when the source is an inline row
    bare = ib.build_bench_header(None, row, "B0")
    assert bare["campaign_mode"] == "full"
    assert bare["seeds"] == [17]


def test_write_bench_plan_pin_verifies_with_driver(tmp_path: Path) -> None:
    """The written plan+pin must satisfy the driver's own verify_plan_pin
    and read_run_plan — i.e. cmd_run would accept this bench root."""
    row = _source_row()
    header = ib.build_bench_header(None, row, "B0")
    plan_path, pin_path = ib.write_bench_plan(tmp_path, row, header)
    assert plan_path.name == "run_plan_v53mp_20k.jsonl"
    assert pin_path.is_file()
    # driver-side plan parse: membership + header rules
    plan = read_run_plan(plan_path)
    assert "v53mp_20k_A_s17" in plan["by_run_id"]
    assert plan["header"]["campaign_mode"] == "full"
    # driver-side pin verification (recomputes plan sha, cross-checks mode)
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    verified = verify_plan_pin(plan_path, pin)
    assert verified["sha256"] == pin["sha256"]
    assert verified["campaign_mode"] == "full"


def test_write_bench_plan_pin_fails_on_byte_tamper(tmp_path: Path) -> None:
    from scripts.agent_ops.ledger.run_ledger import PlanNotFrozenError

    row = _source_row()
    header = ib.build_bench_header(None, row, "B0")
    plan_path, pin_path = ib.write_bench_plan(tmp_path, row, header)
    # flip a byte in the plan AFTER pinning -> verify must refuse
    text = plan_path.read_text(encoding="utf-8")
    plan_path.write_text(text.replace('"arm": "A"', '"arm": "B"'),
                         encoding="utf-8")
    pin = json.loads(pin_path.read_text(encoding="utf-8"))
    with pytest.raises(PlanNotFrozenError):
        verify_plan_pin(plan_path, pin)


def test_validate_bench_root_full_chain(tmp_path: Path) -> None:
    """End-to-end: build a bench root with REAL shas over tmp artifacts and
    run the driver's launch gates (read/pin/stage/five-sha) on it."""
    # tmp artifact tree standing in for the frozen campaign inputs
    config = tmp_path / "campaign.yaml"
    config.write_text("schema: pes2ts_v53mp_campaign_v1\n", encoding="utf-8")
    tables = tmp_path / "data"
    tables.mkdir()
    (tables / "data_manifest.json").write_text('{"files": []}\n',
                                               encoding="utf-8")
    (tables / "teacher_manifest.json").write_text('{"files": []}\n',
                                                  encoding="utf-8")
    split = tables / "split_manifest_v53mp.json"
    split.write_text('{"schema": "v53mp_split_v1"}\n', encoding="utf-8")

    from pes2ts_core.utils.hashing import sha256_file

    row = _source_row()
    row["config_sha256"] = sha256_file(config)
    row["data_sha256"] = sha256_file(tables / "data_manifest.json")
    row["teacher_sha256"] = sha256_file(tables / "teacher_manifest.json")
    row["split_sha256"] = sha256_file(split)
    # features_sha256: the driver hashes the repo's frozen features module
    row["features_sha256"] = sha256_file(
        REPO / "pes2ts_core/pesn/features/key_atom_graph_v3.py")

    header = ib.build_bench_header(None, row, "B0")
    bench_root = tmp_path / "root"
    plan_path, _pin = ib.write_bench_plan(bench_root, row, header)
    result = ib.validate_bench_root(
        bench_root, plan_path, row["run_id"], config, tables, split)
    assert result["five_sha"] == "ok"
    assert result["pin_sha256"]


def test_validate_bench_root_rejects_wrong_sha(tmp_path: Path) -> None:
    config = tmp_path / "campaign.yaml"
    config.write_text("schema: pes2ts_v53mp_campaign_v1\n", encoding="utf-8")
    tables = tmp_path / "data"
    tables.mkdir()
    (tables / "data_manifest.json").write_text('{"files": []}\n',
                                               encoding="utf-8")
    (tables / "teacher_manifest.json").write_text('{"files": []}\n',
                                                  encoding="utf-8")
    split = tables / "split_manifest_v53mp.json"
    split.write_text('{}\n', encoding="utf-8")
    row = _source_row()  # shas are all-zero placeholders -> five-sha fails
    header = ib.build_bench_header(None, row, "B0")
    bench_root = tmp_path / "root"
    plan_path, _pin = ib.write_bench_plan(bench_root, row, header)
    with pytest.raises(SystemExit):
        ib.validate_bench_root(bench_root, plan_path, row["run_id"],
                               config, tables, split)


def test_run_id_rejected_when_not_driver_shaped(tmp_path: Path,
                                                monkeypatch) -> None:
    row = _source_row()
    row["run_id"] = "bench_B0_A_s17"
    args = ib._build_parser().parse_args([
        "--tag", "B0",
        "--run-plan", json.dumps(row),
        "--dry-run", "--reps", "1",
        "--output-root", str(tmp_path / "bench"),
    ])
    with pytest.raises(SystemExit, match="RUN_ID_DRIVER_INVALID"):
        ib.bench_main(args)


def test_load_plan_row_source_file(tmp_path: Path) -> None:
    plan = tmp_path / "run_plan.jsonl"
    plan.write_text(
        json.dumps({"record_type": "plan_header", "tier": "20k",
                    "campaign_mode": "full"}) + "\n"
        + json.dumps({"run_id": "v53mp_20k_A_s17", "tier": "20k",
                      "arm": "A", "seed": 17}) + "\n"
        + json.dumps({"run_id": "v53mp_20k_B_s17", "tier": "20k",
                      "arm": "B", "seed": 17}) + "\n",
        encoding="utf-8")
    row, header, path = ib.load_plan_row(str(plan), None)
    assert row["run_id"] == "v53mp_20k_A_s17"
    assert header["campaign_mode"] == "full"
    assert path == str(plan)
    picked, _, _ = ib.load_plan_row(str(plan), "v53mp_20k_B_s17")
    assert picked["arm"] == "B"
    with pytest.raises(SystemExit):
        ib.load_plan_row(str(plan), "v53mp_20k_Z_s17")


# ---------------------------------------------------------------------------
# dry-run end-to-end
# ---------------------------------------------------------------------------


def test_dry_run_end_to_end_derives_config(tmp_path: Path) -> None:
    """--updates 2000 != driver-effective 15400 -> bench config derived,
    pin accepted by verify_plan_pin, report carries the config fields."""
    out_root = tmp_path / "bench"
    source_cfg = _source_config(tmp_path)
    from pes2ts_core.utils.hashing import sha256_file

    rc = ib.main([
        "--tag", "B0",
        "--run-plan", json.dumps(_source_row()),
        "--config", str(source_cfg),
        "--updates", "2000", "--warmup", "200", "--reps", "2",
        "--output-root", str(out_root),
        "--dry-run",
    ])
    assert rc == 0
    report_path = out_root / "report.json"
    assert report_path.is_file()
    assert (out_root / "report.md").is_file()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == "pes2ts_infra_bench_v2"
    assert report["tag"] == "B0"
    assert report["dry_run"] is True
    assert report["gate_check"] == "pass"     # dry-run skips the subprocess
    assert report["n_reps"] == 2
    assert len(report["reps"]) == 2
    assert report["verdict"]["gate"] == "B0"
    # fabricated stub sits at the B0 target -> verdict must pass
    assert report["verdict"]["pass"] is True
    assert report["aggregate"]["median_ms_update"] == pytest.approx(78.0,
                                                                   abs=1.0)
    # config derivation closed the run-length gap
    assert report["config_source"] == "bench_derived"
    assert report["original_config_sha256"] == sha256_file(source_cfg)
    assert report["driver_effective_updates"] == 2000
    assert report["warnings"] == []          # gap closed -> warning-free
    # per-rep SELF-CONTAINED roots with plan + pin + derived config
    for rep in ("rep0", "rep1"):
        rep_plan = (out_root / rep / "root" / "run_plan_v53mp_20k.jsonl")
        assert rep_plan.is_file(), rep
        pin = json.loads(Path(str(rep_plan) + ".pin.json")
                         .read_text(encoding="utf-8"))
        verify_plan_pin(rep_plan, pin)   # driver-accepted bench root
        # the row binds to the BENCH yaml, and the bench yaml exists
        rep_cfg = out_root / rep / "root" / "config_bench.yaml"
        assert rep_cfg.is_file(), rep
        row = read_run_plan(rep_plan)["by_run_id"]["v53mp_20k_A_s17"]
        assert row["config_sha256"] == sha256_file(rep_cfg)
        assert report["bench_config_sha256"] == sha256_file(rep_cfg)
        # derived yaml actually yields 2000 updates
        assert ib.driver_effective_updates(rep_cfg, "20k") == 2000
        assert (out_root / rep / "root" / "v53mp_20k_A_s17"
                / "train_log.jsonl").is_file()
        assert (out_root / rep / "cgroup_sample.csv").is_file()
    # original yaml untouched
    assert sha256_file(source_cfg) == report["original_config_sha256"]


def test_dry_run_original_config_when_updates_match(tmp_path: Path) -> None:
    """--updates == driver-effective -> original config used directly,
    no derived file, no warnings (the mandated warning-free path)."""
    from pes2ts_core.utils.hashing import sha256_file

    out_root = tmp_path / "bench"
    source_cfg = _source_config(tmp_path)
    # driver-effective for this config/tier is 400 + 1500*10 = 15400
    assert ib.driver_effective_updates(source_cfg, "20k") == 15400
    rc = ib.main([
        "--tag", "B0",
        "--run-plan", json.dumps(dict(
            _source_row(),
            config_sha256=sha256_file(source_cfg))),  # as in the real plan
        "--config", str(source_cfg),
        "--updates", "15400", "--reps", "1",
        "--output-root", str(out_root),
        "--dry-run",
    ])
    assert rc == 0
    report = json.loads((out_root / "report.json").read_text(encoding="utf-8"))
    assert report["config_source"] == "original"
    assert report["bench_config_sha256"] is None
    assert report["driver_effective_updates"] == 15400
    assert report["warnings"] == []
    # no derived file anywhere in the bench tree
    assert not list(out_root.rglob("config_bench.yaml"))
    # plan row keeps the ORIGINAL config sha
    rep_plan = out_root / "rep0" / "root" / "run_plan_v53mp_20k.jsonl"
    row = read_run_plan(rep_plan)["by_run_id"]["v53mp_20k_A_s17"]
    assert row["config_sha256"] == sha256_file(source_cfg)


def test_dry_run_updates_le_freeze_refused(tmp_path: Path) -> None:
    """--updates <= freeze_updates must refuse before anything is written."""
    out_root = tmp_path / "bench"
    with pytest.raises(SystemExit, match="UPDATES_LE_FREEZE"):
        ib.main([
            "--tag", "B0",
            "--run-plan", json.dumps(_source_row()),
            "--config", str(_source_config(tmp_path)),
            "--updates", "400", "--reps", "1",
            "--output-root", str(out_root),
            "--dry-run",
        ])
