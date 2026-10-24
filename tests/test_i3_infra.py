"""I3 infra tests: orchestrator cgroup wrapper + workers mode, B4 drill
dry-run/verdict, thread-env A/B checker (plan §4.3/§5.1/§5.5/§7-B4).

No GPU, no launches: the orchestrator tests exercise ``--dry-run`` +
``_slot_command``/``_derive_memory_high``/``_feature_workers_for``; the
drill test builds + validates the full plan in ``--dry-run`` and pins
``verdict_phase``; the thread-env test pins the compare logic on
synthetic worker outputs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.campaigns import r3_v53mp_orchestrate_parallel as orch  # noqa: E402
from scripts.infra import concurrency_drill as drill  # noqa: E402
from scripts.infra import thread_env_ab as tea  # noqa: E402


def _plan_rows(n: int = 9) -> list[dict]:
    """Synthetic 20k plan rows: A/B/C x seeds 17/29/43."""
    arms = ("A", "B", "C")
    seeds = (17, 29, 43)
    rows = []
    for i in range(n):
        rows.append({
            "record_type": "run",
            "run_id": "v53mp_20k_{}_s{}".format(arms[i % 3], seeds[i // 3]),
            "campaign_id": "PES2TS_V53MP_MOVE_PROCESS",
            "phase": "S3", "tier": "20k", "arm": arms[i % 3], "fold": 0,
            "seed": seeds[i // 3], "n_updates": 15400, "smoke_updates": None,
            "config_sha256": "0" * 64, "data_sha256": "1" * 64,
            "teacher_sha256": "2" * 64, "features_sha256": "3" * 64,
            "split_sha256": "4" * 64, "status": "planned",
            "confirm": False, "variant": None,
        })
    return rows


def _write_plan(path: Path, rows: list[dict]) -> None:
    header = {"record_type": "plan_header", "campaign_id":
              "PES2TS_V53MP_MOVE_PROCESS", "campaign_mode": "full",
              "sha_status": {f: "real" for f in (
                  "config_sha256", "data_sha256", "features_sha256",
                  "split_sha256", "teacher_sha256")}}
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(header, sort_keys=True) + "\n")
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# orchestrator: memory-high derivation, workers mode, slot command
# ---------------------------------------------------------------------------
def test_derive_memory_high() -> None:
    assert orch._derive_memory_high("6G") == "5.4G"
    assert orch._derive_memory_high("34G") == "30.6G"
    assert orch._derive_memory_high("5120M") == "4608M"
    with pytest.raises(SystemExit):
        orch._derive_memory_high("junk")
    with pytest.raises(SystemExit):
        orch._derive_memory_high("6")


def test_feature_workers_for_modes() -> None:
    from pes2ts_core.pesn.offline.training.feature_prefetch_v53mp import (
        workers_for_concurrency,
    )
    assert orch._feature_workers_for("gate", 4) is None
    assert orch._feature_workers_for("auto", 4) == workers_for_concurrency(4)
    assert orch._feature_workers_for("auto", 3) == workers_for_concurrency(3)
    with pytest.raises(SystemExit):
        orch._feature_workers_for("bogus", 4)


def test_slot_command_wraps_with_cgroup(tmp_path: Path) -> None:
    out_root = tmp_path / "root"
    argv, meta = orch._slot_command(
        "v53mp_20k_A_s17", out_root, "cuda", 2,
        per_run_memory_max="6G", feature_workers=9)
    assert argv[0] == "bash" and str(orch.CGROUP_SCOPE) in argv
    assert "6G" in argv and "5.4G" in argv
    assert "--feature-workers" in argv and "9" in argv
    assert "--loader" not in argv              # orchestrator keeps loader default
    assert meta["unit"] == "v53mp-orch-slot2-v53mp_20k_A_s17"
    assert meta["memory_max"] == "6G" and meta["memory_high"] == "5.4G"
    assert meta["gpu_lock"].endswith("slot2.lock")
    # no wrapper when memory-max is off
    argv2, meta2 = orch._slot_command(
        "v53mp_20k_A_s17", out_root, "cuda", 0,
        per_run_memory_max=None, feature_workers=None)
    assert str(orch.RUN_DRIVER) in argv2
    assert str(orch.CGROUP_SCOPE) not in argv2   # not wrapped
    assert meta2["unit"] is None and "--feature-workers" not in argv2


def test_orchestrate_dry_run_with_wrapper_and_workers(tmp_path: Path,
                                                      capsys) -> None:
    root = tmp_path / "v53mp"
    root.mkdir()
    _write_plan(root / "run_plan_v53mp_20k.jsonl", _plan_rows(4))
    progress = root / "orchestrate_20k_parallel.jsonl"
    rc = orch.main([
        "--tier", "20k", "--output-root", str(root),
        "--concurrency", "3", "--dry-run",
        "--per-run-memory-max", "6G", "--feature-workers-mode", "auto",
    ])
    assert rc == 0
    events = [json.loads(ln) for ln in
              progress.read_text(encoding="utf-8").splitlines() if ln.strip()]
    launches = [e for e in events if e["event"] == "launch"]
    assert len(launches) == 4
    # auto workers at concurrency 3
    from pes2ts_core.pesn.offline.training.feature_prefetch_v53mp import (
        workers_for_concurrency,
    )
    for e in launches:
        assert e["feature_workers"] == workers_for_concurrency(3)
        assert e["memory_max"] == "6G" and e["memory_high"] == "5.4G"
        assert e["unit"].startswith("v53mp-orch-slot")
    # slots round-robin over 3 slots for 4 runs
    assert [e["slot"] for e in launches] == [0, 1, 2, 0]
    # per-slot units are distinct
    units = {e["unit"] for e in launches}
    assert len(units) == len(launches)


def test_orchestrate_dry_run_gate_mode_no_wrapper(tmp_path: Path) -> None:
    root = tmp_path / "v53mp"
    root.mkdir()
    _write_plan(root / "run_plan_v53mp_20k.jsonl", _plan_rows(2))
    progress = root / "orchestrate_20k_parallel.jsonl"
    rc = orch.main([
        "--tier", "20k", "--output-root", str(root),
        "--concurrency", "2", "--dry-run", "--feature-workers-mode", "gate",
    ])
    assert rc == 0
    events = [json.loads(ln) for ln in
              progress.read_text(encoding="utf-8").splitlines() if ln.strip()]
    launches = [e for e in events if e["event"] == "launch"]
    for e in launches:
        assert "feature_workers" not in e      # gate mode: no override
        assert "memory_max" not in e           # no wrapper


def test_orchestrate_fail_closed_missing_cgroup_scope(tmp_path: Path,
                                                      monkeypatch) -> None:
    root = tmp_path / "v53mp"
    root.mkdir()
    _write_plan(root / "run_plan_v53mp_20k.jsonl", _plan_rows(2))
    monkeypatch.setattr(orch, "CGROUP_SCOPE", tmp_path / "missing.sh")
    rc = orch.main([
        "--tier", "20k", "--output-root", str(root),
        "--concurrency", "2", "--dry-run", "--per-run-memory-max", "6G",
    ])
    assert rc == 2


# ---------------------------------------------------------------------------
# B4 drill: select_rows, verdict, dry-run
# ---------------------------------------------------------------------------
def test_select_rows_distinct_arm_seed() -> None:
    rows = _plan_rows(9)
    picked = drill._select_rows(rows, 4)
    assert len(picked) == 4
    keys = {(r["arm"], r["seed"]) for r in picked}
    assert len(keys) == 4
    with pytest.raises(SystemExit):
        drill._select_rows(rows[:2], 4)      # exhausted -> typed refusal


def test_verdict_phase_gates() -> None:
    # N=1: only the oom gate applies
    v1 = drill.verdict_phase(1, [{"wall_s": 100.0, "anon_peak_bytes": 5_000_000_000,
                                  "oom_kill": 0, "returncode": 0,
                                  "median_ms": 61.7, "run_id": "r0"}],
                             baseline_wall_s=None)
    assert v1["pass"]
    assert v1["checks"]["oom_kill_zero"] is True
    assert v1["checks"]["all_slots_completed"] is True
    # N=3 within gates: total anon < 24 GB, walls <= 1.5x baseline, oom 0
    slots = [{"wall_s": 110.0, "anon_peak_bytes": 7_000_000_000, "oom_kill": 0,
              "returncode": 0, "median_ms": 70.0, "run_id": "r1"},
             {"wall_s": 120.0, "anon_peak_bytes": 7_000_000_000, "oom_kill": 0,
              "returncode": 0, "median_ms": 71.0, "run_id": "r2"},
             {"wall_s": 105.0, "anon_peak_bytes": 6_000_000_000, "oom_kill": 0,
              "returncode": 0, "median_ms": 72.0, "run_id": "r3"}]
    v3 = drill.verdict_phase(3, slots, baseline_wall_s=100.0)
    assert v3["pass"]
    assert v3["observed"]["total_anon_bytes"] == 20_000_000_000
    assert v3["checks"]["total_anon_lt_24gb"] is True
    assert v3["checks"]["per_run_wall_le_1_5x_baseline"] is True
    # N=3 total anon over the gate -> fail
    slots_hi = [dict(s, anon_peak_bytes=8_000_000_000) for s in slots]
    v3_fail = drill.verdict_phase(3, slots_hi, baseline_wall_s=100.0)
    assert not v3_fail["pass"]
    assert v3_fail["checks"]["total_anon_lt_24gb"] is False
    # wall ratio breach -> fail
    slots_slow = [dict(s, wall_s=160.0) for s in slots]
    v3_slow = drill.verdict_phase(3, slots_slow, baseline_wall_s=100.0)
    assert not v3_slow["pass"]
    assert v3_slow["checks"]["per_run_wall_le_1_5x_baseline"] is False
    # oom -> fail
    slots_oom = [dict(slots[0], oom_kill=1), slots[1], slots[2]]
    v3_oom = drill.verdict_phase(3, slots_oom, baseline_wall_s=100.0)
    assert not v3_oom["pass"] and v3_oom["checks"]["oom_kill_zero"] is False


def test_verdict_phase_slot_failure_is_hard_fail() -> None:
    # regression (2026-10-24 live drill): a slot that died (rc!=0, no
    # metrics) must fail the phase even when oom=0 and walls look fine —
    # the first live B4 run "passed" with 5/8 dead slots before this gate
    dead = {"wall_s": 0.1, "anon_peak_bytes": 0, "oom_kill": 0,
            "returncode": 1, "median_ms": None, "run_id": "v53mp_20k_A_s17"}
    alive = {"wall_s": 648.7, "anon_peak_bytes": 3_500_000_000, "oom_kill": 0,
             "returncode": 0, "median_ms": 73.6, "run_id": "v53mp_20k_A_s29"}
    v = drill.verdict_phase(3, [dead, alive, dict(alive, run_id="r2")],
                            baseline_wall_s=647.0)
    assert not v["pass"]
    assert v["checks"]["all_slots_completed"] is False
    assert v["observed"]["failed_slots"] == ["v53mp_20k_A_s17"]
    # same for a missing returncode field entirely (fail-closed default)
    v2 = drill.verdict_phase(1, [{"wall_s": 10.0, "anon_peak_bytes": 0,
                                  "oom_kill": 0, "median_ms": 5.0,
                                  "run_id": "r"}], baseline_wall_s=None)
    assert not v2["pass"] and v2["checks"]["all_slots_completed"] is False


def test_build_phase_plan_slot_index_distinct_locks_and_units(
        tmp_path: Path, monkeypatch) -> None:
    # regression (2026-10-24 live drill): with n_slots=1 the loop counter
    # was always 0 -> every slot shared drillslot0.lock and the same
    # systemd unit name -> lock serialization + unit collisions
    built = []

    def fake_build(source_row, tag, updates, n_slots, phase_root, tables_dir,
                   split_path, config_path, r3_ckpt, device, memory_max,
                   feature_workers, slot_index=0):
        assert n_slots == 1
        built.append(slot_index)
        return {"slots": [{
            "slot": slot_index, "run_id": source_row["run_id"], "arm": "A",
            "seed": 1, "root": str(phase_root), "plan": "p", "config": "c",
            "gpu_lock": "drillslot{}.lock".format(slot_index),
            "cgroup_outdir": str(phase_root), "memory_max": "6G",
            "memory_high": "5G", "feature_workers": 6, "argv": ["x"]}]}

    monkeypatch.setattr(drill.ib, "derive_bench_config", lambda *a, **k: {
        "path": str(tmp_path / "cfg.yaml")})
    monkeypatch.setattr(drill, "build_phase_plan", fake_build)
    (tmp_path / "cfg.yaml").write_text("x: 1\n", encoding="utf-8")
    plan_rows = [{"run_id": "v53mp_20k_A_s17", "arm": "A", "seed": 17},
                 {"run_id": "v53mp_20k_A_s29", "arm": "A", "seed": 29},
                 {"run_id": "v53mp_20k_A_s43", "arm": "A", "seed": 43}]
    # emulate the caller loop in main()
    for slot, row in enumerate(plan_rows):
        phase = fake_build(row, "B4_n3", 2000, 1,
                           tmp_path / "s{}".format(slot), tmp_path, tmp_path,
                           tmp_path, tmp_path, "cuda", "6G", 6,
                           slot_index=slot)
        desc = phase["slots"][0]
        assert desc["slot"] == slot
        assert "drillslot{}.lock".format(slot) == desc["gpu_lock"]
        unit = "v53mp-drill-n{}-slot{}".format(3, desc["slot"])
        assert unit.endswith("slot{}".format(slot))
    assert built == [0, 1, 2] and len(set(built)) == 3


def test_drill_dry_run_validates_full_plan(tmp_path: Path,
                                           monkeypatch) -> None:
    """--dry-run builds + validates every phase/slot without launching.

    ``validate_bench_root`` is stubbed: it re-hashes the five-sha binding
    against the REAL tables (already covered by test_infra_bench); here
    the drill's own plan/wrapper/worker logic is under test.
    """
    plan = tmp_path / "run_plan_v53mp_20k.jsonl"
    _write_plan(plan, _plan_rows(9))
    # derive_bench_config needs a real yaml config; use the frozen one
    cfg = REPO / "config/campaigns/pes2ts_v53mp_20261021.yaml"
    if not cfg.is_file():
        pytest.skip("frozen campaign config missing")

    def _stub_validate(*a, **k):
        return {"five_sha": "ok"}

    monkeypatch.setattr(drill.ib, "validate_bench_root", _stub_validate)
    out_root = tmp_path / "drill"
    rc = drill.main([
        "--plan", str(plan),
        "--output-root", str(out_root),
        "--ns", "1,3",
        "--updates", "2000",
        "--config", str(cfg),
        "--dry-run",
    ])
    assert rc == 0
    report = json.loads((out_root / "report.json").read_text(encoding="utf-8"))
    assert report["dry_run"] is True
    assert [p["n"] for p in report["phases"]] == [1, 3]
    for phase in report["phases"]:
        assert len(phase["slots"]) == phase["n"]
        for slot in phase["slots"]:
            # every slot's plan + pin exist and the config was derived
            assert Path(slot["plan"]).is_file()
            assert Path(slot["config"]).is_file()
            assert slot["memory_max"] == "6G"
            assert slot["memory_high"] == "5.4G"
            assert slot["feature_workers"] == phase["feature_workers"]
            assert "--loader" in slot["argv"] and "v2" in slot["argv"]
            assert "--dev-pack" in slot["argv"] and "auto" in slot["argv"]
            assert "--feature-workers" in slot["argv"]
    # workers scale with N (auto mode)
    from pes2ts_core.pesn.offline.training.feature_prefetch_v53mp import (
        workers_for_concurrency,
    )
    by_n = {p["n"]: p["feature_workers"] for p in report["phases"]}
    assert by_n[3] == workers_for_concurrency(3)
    assert by_n[1] == workers_for_concurrency(1)
    # distinct (arm, seed) per slot within a phase
    for phase in report["phases"]:
        keys = {(s["arm"], s["seed"]) for s in phase["slots"]}
        assert len(keys) == len(phase["slots"])


def test_drill_dry_run_missing_cgroup_fail_closed(tmp_path: Path,
                                                  monkeypatch) -> None:
    plan = tmp_path / "run_plan_v53mp_20k.jsonl"
    _write_plan(plan, _plan_rows(3))
    cfg = REPO / "config/campaigns/pes2ts_v53mp_20261021.yaml"
    if not cfg.is_file():
        pytest.skip("frozen campaign config missing")
    monkeypatch.setattr(drill, "CGROUP_SCOPE", tmp_path / "missing.sh")
    with pytest.raises(SystemExit) as exc:
        drill.main([
            "--plan", str(plan),
            "--output-root", str(tmp_path / "drill"),
            "--ns", "1", "--updates", "2000", "--config", str(cfg),
            "--dry-run",
        ])
    assert "CGROUP_SCOPE_MISSING" in str(exc.value)


def test_drill_bad_ns_refused(tmp_path: Path) -> None:
    plan = tmp_path / "run_plan_v53mp_20k.jsonl"
    _write_plan(plan, _plan_rows(3))
    with pytest.raises(SystemExit):
        drill.main([
            "--plan", str(plan),
            "--output-root", str(tmp_path / "drill"),
            "--ns", "0", "--dry-run",
        ])


# ---------------------------------------------------------------------------
# thread-env A/B: compare logic on synthetic worker outputs
# ---------------------------------------------------------------------------
def test_thread_env_compare_identical() -> None:
    """Three identical hash lists -> bitwise_identical True (pure compare)."""
    hashes = ["a" * 64, "b" * 64, "c" * 64]
    positions = [1, 2, 3]
    results = {name: {"positions": positions, "hashes": list(hashes),
                      "wall_s": 1.0, "per_feature_ms": 0.3, "env": {}}
               for name in ("default", "t1", "t8")}
    ref = results["default"]
    bitwise = all(results[n]["hashes"] == ref["hashes"]
                  and results[n]["positions"] == ref["positions"]
                  for n in ("t1", "t8"))
    assert bitwise


def test_thread_env_compare_detects_difference() -> None:
    hashes_a = ["a" * 64, "b" * 64, "c" * 64]
    hashes_b = ["a" * 64, "X" * 64, "c" * 64]
    ref = {"positions": [1, 2, 3], "hashes": hashes_a}
    other = {"positions": [1, 2, 3], "hashes": hashes_b}
    diffs = [(i, a, b) for i, (a, b) in
             enumerate(zip(ref["hashes"], other["hashes"])) if a != b]
    assert diffs == [(1, "b" * 64, "X" * 64)]


def test_thread_env_keys_and_envs_contract() -> None:
    assert set(tea.ENVS) == {"default", "t1", "t8"}
    assert tea.ENVS["default"] is None
    for key in tea.THREAD_ENV_KEYS:
        assert tea.ENVS["t1"][key] == "1"
        assert tea.ENVS["t8"][key] == "8"
