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


def test_verdict_b0_seconds_target_unit_fix() -> None:
    """--b0-ms-target 0.078 (s/update, plan §7 wording) must give the same
    [66.3, 89.7] ms band as 78.0 — not a nonsense [0.07, 0.09] ms band."""
    secs = ib.verdict_b0(median_ms=78.0, anon_peak_bytes=int(26.0e9),
                         swap_max_bytes=0, oom_kill=0, ms_target=0.078)
    millis = ib.verdict_b0(median_ms=78.0, anon_peak_bytes=int(26.0e9),
                           swap_max_bytes=0, oom_kill=0, ms_target=78.0)
    assert secs["pass"] is True and millis["pass"] is True
    assert secs["bounds"]["median_ms_update"] == [66.3, 89.7]
    assert secs["bounds"] == millis["bounds"]
    # an observed 98.8 ms (the archived B0) is OUTSIDE the 78±15% band
    # either way — the fix is about the band's unit, not its width
    slow = ib.verdict_b0(median_ms=98.795, anon_peak_bytes=int(26.0e9),
                         swap_max_bytes=0, oom_kill=0, ms_target=0.078)
    assert slow["checks"]["median_ms_update_in_band"] is False
    assert slow["bounds"]["median_ms_update"] == [66.3, 89.7]
    # legacy ms-scale call sites are untouched
    assert ib.normalize_ms_target(78.0) == 78.0
    assert ib.normalize_ms_target(0.078) == 78.0
    assert ib.normalize_ms_target(0.5) == 500.0


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


# ---------------------------------------------------------------------------
# per-rung / ladder recompute (read-only, no GPU, no re-run)
# ---------------------------------------------------------------------------


def _rep(rep: int, median_ms: float, anon_gb: float, startup_s: float,
         swap: int = 0, oom: int = 0) -> dict:
    return {"rep": rep, "median_ms": median_ms, "p95_ms": median_ms + 3.0,
            "startup_s": startup_s, "anon_peak_bytes": int(anon_gb * 1e9),
            "swap_max_bytes": swap, "oom_kill": oom, "vram_peak_mib": 2500,
            "exit": 0, "gate_exit": 0}


def _report(tag: str, reps: list, original_pass: bool = False) -> dict:
    """A minimal archived-report-shaped dict (the recompute only needs
    tag / reps / verdict; aggregate is derived from reps)."""
    return {"schema": "pes2ts_infra_bench_v2", "tag": tag,
            "generated_at": "2026-10-24T00:00:00Z", "reps": reps,
            "verdict": {"gate": tag, "pass": original_pass,
                        "checks": {}, "bounds": {}, "observed": {}}}


#: the four archived bench reports' per-rep metrics (B0_20261023T222828Z ..
#: B3_20261024T000312Z), mirrored so the ladder tests do not depend on
#: outputs/ being present.
_ARCHIVED_LADDER = [
    ("B0", [_rep(0, 98.795, 30.785, 264.7), _rep(1, 96.9, 30.798, 289.1),
            _rep(2, 99.8, 30.857, 286.0)]),
    ("B1", [_rep(0, 73.405, 7.193, 25.1), _rep(1, 75.28, 7.202, 24.9),
            _rep(2, 77.115, 7.205, 24.1)]),
    ("B2", [_rep(0, 77.145, 3.539, 25.4), _rep(1, 74.455, 3.59, 26.0),
            _rep(2, 75.48, 3.537, 26.0)]),
    ("B3", [_rep(0, 87.6, 3.566, 26.1), _rep(1, 92.575, 3.618, 25.6),
            _rep(2, 89.73, 3.557, 25.8)]),
]


def _archived_ladder_reports() -> list:
    return [_report(tag, reps) for tag, reps in _ARCHIVED_LADDER]


def test_ladder_b0_self_certifies() -> None:
    """B0 defines its own baseline (median ms / max anon from its own
    reps); the archived 98.8 ms / 30.9 GB run is green against itself."""
    b0 = _archived_ladder_reports()[0]
    verdict = ib.verdict_b0_self(b0["reps"])
    assert verdict["pass"] is True
    assert verdict["mode"] == "self_baseline"
    assert verdict["failing_checks"] == []
    assert verdict["baseline"]["median_ms_update"] == 98.795
    assert verdict["baseline"]["anon_peak_gb"] == 30.857
    # per-rep band: baseline ±15%
    assert verdict["bounds"]["median_ms_update"] == [83.98, 113.61]
    assert verdict["bounds"]["anon_peak_gb"] == [26.23, 35.49]
    assert all(c["median_ms_in_band"] and c["anon_peak_in_band"]
               for c in verdict["observed"]["per_rep"])


def test_ladder_b0_self_fails_on_outlier_rep() -> None:
    """One rep outside the self-band (or with swap/oom) red-flags B0."""
    reps = [_rep(0, 98.795, 30.8, 286.0), _rep(1, 96.9, 30.8, 286.0),
            _rep(2, 150.0, 30.8, 286.0)]   # 150 > 98.795*1.15
    verdict = ib.verdict_b0_self(reps)
    assert verdict["pass"] is False
    assert "reps_median_ms_in_band" in verdict["failing_checks"]
    # swap/oom stay hard gates inside the self-band too
    swapped = ib.verdict_b0_self(
        [_rep(0, 98.0, 30.8, 286.0), _rep(1, 97.0, 30.8, 286.0, swap=4096),
         _rep(2, 99.0, 30.8, 286.0)])
    assert swapped["pass"] is False
    assert "swap_gate" in swapped["failing_checks"]
    killed = ib.verdict_b0_self(
        [_rep(0, 98.0, 30.8, 286.0), _rep(1, 97.0, 30.8, 286.0),
         _rep(2, 99.0, 30.8, 286.0, oom=1)])
    assert killed["pass"] is False
    assert "oom_gate" in killed["failing_checks"]


def test_ladder_b1_b2_green_b3_red_only_ms() -> None:
    """Regression-only policy: B1/B2 improve massively and are green (no
    lower band); B3 is red ONLY because 89.73 > 1.15 x 75.48 = 86.80."""
    reports = _archived_ladder_reports()
    result = ib.recompute_ladder(reports)
    rungs = {r["tag"]: r for r in result["rungs"]}
    assert [r["tag"] for r in result["rungs"]] == ["B0", "B1", "B2", "B3"]
    assert result["pass"] is False
    # B0 self-baseline
    assert rungs["B0"]["verdict"]["pass"] is True
    assert rungs["B0"]["baseline_tag"] is None
    # B1: 75.28 <= 1.15*98.795 = 113.61, anon 7.205 <= 35.49
    b1 = rungs["B1"]["verdict"]
    assert b1["pass"] is True and b1["failing_checks"] == []
    assert b1["bounds"]["median_ms_update_max"] == 113.61
    assert b1["bounds"]["anon_peak_gb_max"] == 35.49
    assert b1["observed"]["median_ms"] == 75.28
    assert b1["observed"]["anon_peak_gb"] == 7.205
    # B2 vs B1: 75.48 <= 86.57, anon 3.59 <= 8.29
    b2 = rungs["B2"]["verdict"]
    assert b2["pass"] is True and b2["failing_checks"] == []
    assert b2["bounds"]["median_ms_update_max"] == 86.57
    assert b2["bounds"]["anon_peak_gb_max"] == 8.29
    # B3 vs B2: 89.73 > 86.80 -> red for ms ONLY (anon 3.618 <= 4.13 green)
    b3 = rungs["B3"]["verdict"]
    assert b3["pass"] is False
    assert b3["failing_checks"] == ["median_ms_no_regression"]
    assert b3["checks"]["median_ms_no_regression"] is False
    assert b3["checks"]["anon_no_regression"] is True
    assert b3["checks"]["startup_no_regression"] is True
    assert b3["checks"]["swap_gate"] is True
    assert b3["checks"]["oom_gate"] is True
    assert b3["bounds"]["median_ms_update_max"] == 86.8
    assert b3["observed"]["median_ms"] == 89.73
    # the ORIGINAL verdict of each archived report is preserved for audit
    for report, rung in zip(reports, result["rungs"]):
        assert rung["original_verdict"] == report["verdict"]
        assert rung["original_verdict"]["pass"] is False  # old fixed band


def test_ladder_allows_large_improvement_no_lower_band() -> None:
    """A rung 10x faster / 10x leaner than the one below is green."""
    base = _report("B0", [_rep(i, 100.0, 30.0, 100.0) for i in range(3)])
    lean = _report("B1", [_rep(i, 10.0, 3.0, 10.0) for i in range(3)])
    result = ib.recompute_ladder([base, lean])
    assert result["rungs"][1]["verdict"]["pass"] is True
    assert result["pass"] is True


def test_ladder_swap_oom_still_hard_fail() -> None:
    """Even a big improvement is red when swap or oom is nonzero."""
    base = _report("B0", [_rep(i, 98.0, 30.0, 260.0) for i in range(3)])
    swapped = _report("B1", [_rep(0, 70.0, 7.0, 25.0, swap=1 << 20),
                             _rep(1, 71.0, 7.0, 25.0),
                             _rep(2, 72.0, 7.0, 25.0)])
    result = ib.recompute_ladder([base, swapped])
    b1 = result["rungs"][1]["verdict"]
    assert b1["pass"] is False
    assert b1["failing_checks"] == ["swap_gate"]
    assert b1["checks"]["median_ms_no_regression"] is True
    assert b1["checks"]["anon_no_regression"] is True
    # oom alone is enough too
    oomy = _report("B1", [_rep(0, 70.0, 7.0, 25.0),
                          _rep(1, 71.0, 7.0, 25.0, oom=2),
                          _rep(2, 72.0, 7.0, 25.0)])
    b1_oom = ib.recompute_ladder([base, oomy])["rungs"][1]["verdict"]
    assert b1_oom["pass"] is False
    assert b1_oom["failing_checks"] == ["oom_gate"]


def test_ladder_startup_regression_red() -> None:
    """startup <= 1.15 x previous rung's startup is part of the ladder."""
    base = _report("B0", [_rep(i, 98.0, 30.0, 25.0) for i in range(3)])
    slow_start = _report("B1", [_rep(0, 70.0, 7.0, 40.0),
                                _rep(1, 71.0, 7.0, 41.0),
                                _rep(2, 72.0, 7.0, 39.0)])
    b1 = ib.recompute_ladder([base, slow_start])["rungs"][1]["verdict"]
    assert b1["pass"] is False
    assert b1["failing_checks"] == ["startup_no_regression"]
    assert b1["bounds"]["startup_s_max"] == 28.75


def test_ladder_requires_b0_baseline() -> None:
    """A ladder not starting at B0 is refused (fail-closed)."""
    with pytest.raises(SystemExit, match="RECOMPUTE_BASELINE_MISSING"):
        ib.recompute_ladder([_report("B1", [_rep(0, 75.0, 7.0, 25.0)])])
    with pytest.raises(SystemExit, match="RECOMPUTE_EMPTY"):
        ib.recompute_ladder([])


# ---- CLI --recompute over the real archived reports ----------------------


ARCHIVED_BENCH_ROOT = REPO / "outputs/pesn/v53mp/infra_bench"
ARCHIVED_TAGS = ("B0_20261023T222828Z", "B1_20261023T230605Z",
                 "B2_20261023T233021Z", "B3_20261024T000312Z")


def _archived_report_paths() -> list:
    return [ARCHIVED_BENCH_ROOT / tag / "report.json" for tag in ARCHIVED_TAGS]


def test_recompute_archived_reports_json(tmp_path: Path) -> None:
    """End-to-end: recompute the four ARCHIVED B0..B3 report.json files —
    B0/B1/B2 green, B3 red for the ms regression only — without touching
    the originals."""
    paths = _archived_report_paths()
    if not all(p.is_file() for p in paths):
        pytest.skip("archived B0-B3 report.json not present")
    before = {p: p.read_bytes() for p in paths}
    out = tmp_path / "recompute.json"
    rc = ib.main(["--recompute"] + [str(p) for p in paths]
                 + ["--recompute-out", str(out)])
    assert rc == 1                      # B3 ms regression -> overall red
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["schema"] == ib.RECOMPUTE_SCHEMA
    assert result["kind"] == "ladder"
    assert result["pass"] is False
    rungs = {r["tag"]: r for r in result["rungs"]}
    assert [r["tag"] for r in result["rungs"]] == ["B0", "B1", "B2", "B3"]
    assert rungs["B0"]["verdict"]["pass"] is True
    assert rungs["B1"]["verdict"]["pass"] is True
    assert rungs["B2"]["verdict"]["pass"] is True
    assert rungs["B3"]["verdict"]["pass"] is False
    assert rungs["B3"]["verdict"]["failing_checks"] == [
        "median_ms_no_regression"]
    assert rungs["B3"]["verdict"]["checks"]["anon_no_regression"] is True
    # the archived metrics are the ones the acceptance quotes
    assert rungs["B0"]["verdict"]["baseline"]["median_ms_update"] == 98.795
    assert rungs["B0"]["verdict"]["baseline"]["anon_peak_gb"] == 30.857
    assert rungs["B1"]["verdict"]["observed"]["median_ms"] == 75.28
    assert rungs["B1"]["verdict"]["observed"]["anon_peak_gb"] == 7.205
    assert rungs["B2"]["verdict"]["observed"]["median_ms"] == 75.48
    assert rungs["B3"]["verdict"]["observed"]["median_ms"] == 89.73
    # sources recorded + ORIGINAL verdicts preserved for audit
    assert [s["tag"] for s in result["sources"]] == ["B0", "B1", "B2", "B3"]
    for source, rung in zip(result["sources"], result["rungs"]):
        assert source["original_verdict"] == rung["original_verdict"]
        assert rung["original_verdict"]["pass"] is False  # old fixed band
    # archived reports byte-for-byte untouched
    for p, data in before.items():
        assert p.read_bytes() == data, p


def test_recompute_accepts_bench_root_dir_and_prints_stdout(
        tmp_path: Path, capsys) -> None:
    """--recompute accepts a bench ROOT dir (report.json inferred) and
    emits a single pure JSON document on stdout."""
    paths = _archived_report_paths()
    if not all(p.is_file() for p in paths):
        pytest.skip("archived B0-B3 report.json not present")
    dirs = [p.parent for p in paths]
    rc = ib.main(["--recompute"] + [str(d) for d in dirs])
    assert rc == 1
    captured = capsys.readouterr()
    result = json.loads(captured.out)   # stdout must be ONE JSON doc
    assert result["kind"] == "ladder"
    assert [r["tag"] for r in result["rungs"]] == ["B0", "B1", "B2", "B3"]
    # the one-line summary went to stderr
    summary = json.loads(captured.err.splitlines()[0])
    assert summary["action"] == "infra_bench_recompute"
    assert summary["pass"] is False


def test_recompute_cli_on_synthetic_reports_and_out_refusal(
        tmp_path: Path) -> None:
    """--recompute works on synthetic archived reports too; writing the
    verdict over an INPUT report is refused."""
    paths = []
    for tag, reps in _ARCHIVED_LADDER:
        p = tmp_path / tag / "report.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        doc = _report(tag, reps)
        p.write_text(json.dumps(doc), encoding="utf-8")
        paths.append(p)
    # green ladder: swap B3 for a non-regressing rung
    good_b3 = _report("B3", [_rep(i, 74.0, 3.5, 25.5) for i in range(3)])
    paths[-1].write_text(json.dumps(good_b3), encoding="utf-8")
    out = tmp_path / "ladder.json"
    rc = ib.main(["--recompute"] + [str(p) for p in paths]
                 + ["--recompute-out", str(out)])
    assert rc == 0
    result = json.loads(out.read_text(encoding="utf-8"))
    assert result["pass"] is True
    # refusal: --recompute-out pointing at an input report
    with pytest.raises(SystemExit, match="RECOMPUTE_OUT_IS_INPUT"):
        ib.main(["--recompute", str(paths[0]), "--recompute-out",
                 str(paths[0])])
    # missing input
    with pytest.raises(SystemExit, match="RECOMPUTE_REPORT_MISSING"):
        ib.main(["--recompute", str(tmp_path / "nope" / "report.json")])
    # original synthetic reports untouched by the green recompute
    assert json.loads(paths[-1].read_text(encoding="utf-8"))["reps"] == \
        good_b3["reps"]


# ---------------------------------------------------------------------------
# gate-check subprocess: cgroup-contained (mocked Popen; no systemd/GPU)
# ---------------------------------------------------------------------------


_GATE_OK_JSON = json.dumps(
    {"items": {"config_yaml": "pass"}, "details": {}, "ok": True}) + "\n"
_GATE_WRAPPER_STDERR = (
    "[cgroup_scope] fallback=0 unit=infra-scope-1 max=8G high=7G "
    "outdir=/x/_gate_check\n"
    '{"peak_anon_bytes":123,"max_swap_current_bytes":0,'
    '"swap_sustained_s":0,"oom_kill":0,"cmd_exit":0,"gate_exit":0}\n'
)


def _gate_outdir_from_cmd(cmd: list) -> Path:
    i = 2                                   # after bash + cgroup_scope.sh
    if cmd[i] == "--fallback":
        i += 1
    return Path(cmd[i + 2])


class _FakeGateProc:
    """subprocess.Popen stand-in for the gate-check wrapper launch."""

    def __init__(self, cmd, rc=0, stdout_text="", stderr_text="",
                 summary=None, timeout_on_first_wait=False, calls=None):
        self.cmd = list(cmd)
        self.returncode = None
        self.pid = 4242
        self._rc = rc
        self._waits = 0
        self._timeout_on_first_wait = timeout_on_first_wait
        if calls is not None:
            calls.append(self)
        outdir = _gate_outdir_from_cmd(self.cmd)
        if summary is not None:
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "cgroup_summary.json").write_text(
                json.dumps(summary), encoding="utf-8")

    def wait(self, timeout=None):
        self._waits += 1
        if self._timeout_on_first_wait and self._waits == 1:
            raise ib.subprocess.TimeoutExpired(cmd=self.cmd,
                                               timeout=timeout or 0)
        self.returncode = self._rc
        return self.returncode


def _install_fake_gate_popen(monkeypatch, rc=0, stdout_text="",
                             stderr_text="", summary=None,
                             timeout_on_first_wait=False):
    """Patch ib.subprocess.Popen so run_gate_check never spawns anything."""
    created = []

    def _popen(cmd, **kwargs):
        proc = _FakeGateProc(cmd, rc=rc, summary=summary,
                             timeout_on_first_wait=timeout_on_first_wait,
                             calls=created)
        proc.kwargs = kwargs
        out_fh = kwargs.get("stdout")
        err_fh = kwargs.get("stderr")
        if stdout_text and out_fh is not None:
            out_fh.write(stdout_text)
        if stderr_text and err_fh is not None:
            err_fh.write(stderr_text)
        return proc

    monkeypatch.setattr(ib.subprocess, "Popen", _popen)
    return created


def test_run_gate_check_wraps_driver_in_cgroup_scope(
        tmp_path: Path, monkeypatch) -> None:
    """The gate-check driver must run INSIDE cgroup_scope.sh with the
    8G/7G defaults, outdir under the probe root, and its own session
    (so a timeout can kill the whole tree)."""
    created = _install_fake_gate_popen(
        monkeypatch, rc=0, stdout_text=_GATE_OK_JSON,
        stderr_text=_GATE_WRAPPER_STDERR, summary={"cmd_exit": 0})
    probe = tmp_path / "bench"
    result = ib.run_gate_check(tmp_path / "cfg.yaml", tmp_path / "tables",
                               None, probe)
    assert len(created) == 1
    cmd = created[0].cmd
    assert cmd[0] == "bash"
    assert cmd[1] == str(ib.CGROUP_SCOPE)
    assert cmd[2:5] == ["8G", "7G", str(probe / "_gate_check")]
    assert cmd[5] == sys.executable
    assert cmd[6] == str(ib.RUN_DRIVER)
    assert cmd[7:9] == ["--output-root", str(probe)]
    assert cmd[9] == "gate-check"
    assert "--config" in cmd and "--tables-dir" in cmd
    kwargs = created[0].kwargs
    assert kwargs["start_new_session"] is True
    assert kwargs["cwd"] == str(ib.REPO)
    # success propagation
    assert result["ok"] is True
    assert result["returncode"] == 0
    assert result["timed_out"] is False
    assert result["items"] == {"config_yaml": "pass"}
    # stdout/stderr preserved under bench_top/_gate_check/
    gate_dir = probe / "_gate_check"
    assert (gate_dir / "gate_stdout.log").read_text(
        encoding="utf-8") == _GATE_OK_JSON
    assert "[cgroup_scope]" in (gate_dir / "gate_stderr.log").read_text(
        encoding="utf-8")
    assert (gate_dir / "cgroup_summary.json").is_file()


def test_run_gate_check_split_flag_forwarded(
        tmp_path: Path, monkeypatch) -> None:
    created = _install_fake_gate_popen(
        monkeypatch, rc=0, stdout_text=_GATE_OK_JSON)
    ib.run_gate_check(tmp_path / "cfg.yaml", tmp_path / "tables",
                      tmp_path / "split.json", tmp_path / "bench")
    cmd = created[0].cmd
    assert cmd[cmd.index("--split") + 1] == str(tmp_path / "split.json")


def test_run_gate_check_custom_memory_limits(
        tmp_path: Path, monkeypatch) -> None:
    created = _install_fake_gate_popen(
        monkeypatch, rc=0, stdout_text=_GATE_OK_JSON)
    ib.run_gate_check(tmp_path / "cfg.yaml", tmp_path / "tables", None,
                      tmp_path / "bench", gate_memory_max="4G",
                      gate_memory_high="3G")
    cmd = created[0].cmd
    assert cmd[2:5] == ["4G", "3G", str(tmp_path / "bench" / "_gate_check")]


def test_run_gate_check_fallback_flag_forwarded(
        tmp_path: Path, monkeypatch) -> None:
    created = _install_fake_gate_popen(
        monkeypatch, rc=0, stdout_text=_GATE_OK_JSON)
    ib.run_gate_check(tmp_path / "cfg.yaml", tmp_path / "tables", None,
                      tmp_path / "bench", fallback=True)
    cmd = created[0].cmd
    assert cmd[2] == "--fallback"
    assert _gate_outdir_from_cmd(cmd) == tmp_path / "bench" / "_gate_check"


def test_run_gate_check_wrapper_stderr_does_not_pollute_json(
        tmp_path: Path, monkeypatch) -> None:
    """cgroup_scope.sh logs (incl. its own JSON summary line) go to
    stderr; parsing must pick the DRIVER's JSON off stdout only."""
    _install_fake_gate_popen(
        monkeypatch, rc=0, stdout_text=_GATE_OK_JSON,
        stderr_text=_GATE_WRAPPER_STDERR)
    result = ib.run_gate_check(tmp_path / "cfg.yaml", tmp_path / "tables",
                               None, tmp_path / "bench")
    assert result["ok"] is True
    assert result["items"] == {"config_yaml": "pass"}
    assert result["stderr_tail"].startswith("[cgroup_scope]")


def test_run_gate_check_driver_failure_never_becomes_success(
        tmp_path: Path, monkeypatch) -> None:
    """rc != 0 (driver gate failure OR wrapper oom/swap gate 90/91) must
    stay a failure even when the stdout JSON claims ok — the wrapper may
    only tighten, never loosen, the verdict."""
    for rc in (1, 91):
        _install_fake_gate_popen(
            monkeypatch, rc=rc, stdout_text=_GATE_OK_JSON,
            summary={"cmd_exit": rc})
        result = ib.run_gate_check(tmp_path / "cfg.yaml", tmp_path / "tables",
                                   None, tmp_path / "bench")
        assert result["ok"] is False
        assert result["returncode"] == rc
        monkeypatch.undo()


def test_run_gate_check_driver_ok_false_is_failure(
        tmp_path: Path, monkeypatch) -> None:
    _install_fake_gate_popen(
        monkeypatch, rc=1,
        stdout_text=json.dumps({"items": {"x": "fail"}, "ok": False}) + "\n")
    result = ib.run_gate_check(tmp_path / "cfg.yaml", tmp_path / "tables",
                               None, tmp_path / "bench")
    assert result["ok"] is False
    assert result["items"] == {"x": "fail"}


def test_run_gate_check_wrapper_summary_cmd_exit_blocks_success(
        tmp_path: Path, monkeypatch) -> None:
    """rc==0 + driver ok, but the wrapper summary reports cmd_exit!=0:
    the gate must fail closed."""
    _install_fake_gate_popen(
        monkeypatch, rc=0, stdout_text=_GATE_OK_JSON,
        summary={"cmd_exit": 137})
    result = ib.run_gate_check(tmp_path / "cfg.yaml", tmp_path / "tables",
                               None, tmp_path / "bench")
    assert result["ok"] is False


def test_run_gate_check_timeout_kills_process_tree(
        tmp_path: Path, monkeypatch) -> None:
    created = _install_fake_gate_popen(
        monkeypatch, rc=-9, timeout_on_first_wait=True)
    killed = []
    monkeypatch.setattr(ib.os, "getpgid", lambda pid: 4343)
    monkeypatch.setattr(ib.os, "killpg",
                        lambda pgid, sig: killed.append((pgid, sig)))
    result = ib.run_gate_check(tmp_path / "cfg.yaml", tmp_path / "tables",
                               None, tmp_path / "bench")
    assert killed == [(4343, ib.signal.SIGKILL)]
    assert result["ok"] is False
    assert result["timed_out"] is True
    assert result["returncode"] == -9
    # both waits happened: the timed one, then the post-kill reap
    assert created[0]._waits == 2


def test_run_gate_check_unparseable_stdout_fails_closed(
        tmp_path: Path, monkeypatch) -> None:
    _install_fake_gate_popen(monkeypatch, rc=0, stdout_text="garbage\n")
    result = ib.run_gate_check(tmp_path / "cfg.yaml", tmp_path / "tables",
                               None, tmp_path / "bench")
    assert result["ok"] is False       # rc 0 but no driver ok -> failure


def test_gate_memory_cli_defaults_and_overrides() -> None:
    args = ib._build_parser().parse_args([])
    assert args.gate_memory_max == "8G"
    assert args.gate_memory_high == "7G"
    args = ib._build_parser().parse_args([
        "--gate-memory-max", "4G", "--gate-memory-high", "2G"])
    assert args.gate_memory_max == "4G"
    assert args.gate_memory_high == "2G"


def test_bench_main_passes_gate_memory_to_gate_check(
        tmp_path: Path, monkeypatch) -> None:
    """CLI overrides flow from bench_main into run_gate_check kwargs."""
    captured = {}

    def _fake_gate(config_path, tables_dir, split_path, probe_root,
                   **kwargs):
        captured.update(kwargs)
        captured["probe_root"] = probe_root
        return {"ok": False, "returncode": 1, "items": None}

    monkeypatch.setattr(ib, "run_gate_check", _fake_gate)
    rc = ib.main([
        "--tag", "B0",
        "--run-plan", json.dumps(_source_row()),
        "--config", str(_source_config(tmp_path)),
        "--updates", "2000", "--reps", "1",
        "--output-root", str(tmp_path / "bench"),
        "--gate-memory-max", "6G", "--gate-memory-high", "5G",
    ])
    assert rc == 2                       # gate failure -> exit 2
    assert captured["gate_memory_max"] == "6G"
    assert captured["gate_memory_high"] == "5G"
    assert captured["probe_root"] == tmp_path / "bench"
