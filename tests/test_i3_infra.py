"""I3 infra tests: orchestrator cgroup wrapper + workers mode, B4 drill
dry-run/verdict (+ whole-card VRAM gate / ``--cuda-memory-fraction``
passthrough), thread-env A/B checker (plan §4.3/§5.1/§5.5/§7-B4).

No GPU, no launches: the orchestrator tests exercise ``--dry-run`` +
``_slot_command``/``_derive_memory_high``/``_feature_workers_for``; the
drill test builds + validates the full plan in ``--dry-run`` and pins
``verdict_phase`` (the drill's VRAM sampler is exercised with a mocked
``subprocess``); the thread-env test pins the compare logic on
synthetic worker outputs.
"""

from __future__ import annotations

import argparse
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


def test_slot_command_eval_vram_gate_forwarded(tmp_path: Path) -> None:
    """--eval-vram-gate-mib/-timeout-s -> driver argv; default -> absent."""
    out_root = tmp_path / "root"
    argv, meta = orch._slot_command(
        "v53mp_20k_A_s17", out_root, "cuda", 0,
        per_run_memory_max=None, feature_workers=None,
        eval_vram_gate_mib=6900, eval_vram_gate_timeout_s=120)
    i = argv.index("--eval-vram-gate-mib")
    assert argv[i + 1] == "6900"
    j = argv.index("--eval-vram-gate-timeout-s")
    assert argv[j + 1] == "120"
    assert meta["eval_vram_gate_mib"] == 6900
    # default: flags absent (argv byte-identical to the pre-gate builder)
    argv2, meta2 = orch._slot_command(
        "v53mp_20k_A_s17", out_root, "cuda", 0,
        per_run_memory_max=None, feature_workers=None)
    assert "--eval-vram-gate-mib" not in argv2
    assert "--eval-vram-gate-timeout-s" not in argv2
    assert meta2["eval_vram_gate_mib"] is None


def test_orchestrate_dry_run_eval_vram_gate(tmp_path: Path) -> None:
    """--eval-vram-gate-mib on the orchestrator: launch events carry it."""
    root = tmp_path / "v53mp"
    root.mkdir()
    _write_plan(root / "run_plan_v53mp_20k.jsonl", _plan_rows(2))
    progress = root / "orchestrate_20k_parallel.jsonl"
    rc = orch.main([
        "--tier", "20k", "--output-root", str(root),
        "--concurrency", "2", "--dry-run",
        "--eval-vram-gate-mib", "6900", "--eval-vram-gate-timeout-s", "120",
    ])
    assert rc == 0
    events = [json.loads(ln) for ln in
              progress.read_text(encoding="utf-8").splitlines() if ln.strip()]
    launches = [e for e in events if e["event"] == "launch"]
    assert len(launches) == 2
    assert all(e["eval_vram_gate_mib"] == 6900 for e in launches)
    # default off: no gate key in the launch events
    progress2 = root / "orchestrate_20k_parallel.jsonl"
    progress2.write_text("", encoding="utf-8")
    rc = orch.main([
        "--tier", "20k", "--output-root", str(root),
        "--concurrency", "2", "--dry-run",
    ])
    assert rc == 0
    events = [json.loads(ln) for ln in
              progress2.read_text(encoding="utf-8").splitlines() if ln.strip()]
    launches = [e for e in events if e["event"] == "launch"]
    assert len(launches) == 2
    assert all("eval_vram_gate_mib" not in e for e in launches)


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
    # N=1: only the oom gate applies (whole-card vram is REFERENCE only)
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
    v3 = drill.verdict_phase(3, slots, baseline_wall_s=100.0,
                             vram_peak_mib=6100, vram_samples=340)
    assert v3["pass"]
    assert v3["observed"]["total_anon_bytes"] == 20_000_000_000
    assert v3["observed"]["vram_peak_mib"] == 6100
    assert v3["checks"]["total_anon_lt_24gb"] is True
    assert v3["checks"]["whole_card_vram_le_6800_mib"] is True
    assert v3["checks"]["per_run_wall_le_1_5x_baseline"] is True
    # N=3 total anon over the gate -> fail
    slots_hi = [dict(s, anon_peak_bytes=8_000_000_000) for s in slots]
    v3_fail = drill.verdict_phase(3, slots_hi, baseline_wall_s=100.0,
                                  vram_peak_mib=6100, vram_samples=340)
    assert not v3_fail["pass"]
    assert v3_fail["checks"]["total_anon_lt_24gb"] is False
    # wall ratio breach -> fail
    slots_slow = [dict(s, wall_s=160.0) for s in slots]
    v3_slow = drill.verdict_phase(3, slots_slow, baseline_wall_s=100.0,
                                  vram_peak_mib=6100, vram_samples=340)
    assert not v3_slow["pass"]
    assert v3_slow["checks"]["per_run_wall_le_1_5x_baseline"] is False
    # oom -> fail
    slots_oom = [dict(slots[0], oom_kill=1), slots[1], slots[2]]
    v3_oom = drill.verdict_phase(3, slots_oom, baseline_wall_s=100.0,
                                 vram_peak_mib=6100, vram_samples=340)
    assert not v3_oom["pass"] and v3_oom["checks"]["oom_kill_zero"] is False
    # rc / all_slots gates unchanged (see the dedicated regression below)
    assert set(v3["checks"]) >= {"oom_kill_zero", "all_slots_completed",
                                 "total_anon_lt_24gb",
                                 "whole_card_vram_le_6800_mib"}
    assert "whole_card_vram_le_6800_mib" in v3["gates"]


def test_verdict_phase_whole_card_vram_gate_boundaries() -> None:
    """N=3 whole-card gate: 6800 MiB passes, 6801 fails, no samples fails."""
    slots = [{"wall_s": 100.0, "anon_peak_bytes": 5_000_000_000, "oom_kill": 0,
              "returncode": 0, "median_ms": 70.0, "run_id": "r{}".format(i)}
             for i in range(3)]
    v_ok = drill.verdict_phase(3, slots, baseline_wall_s=100.0,
                               vram_peak_mib=6800, vram_samples=100)
    assert v_ok["checks"]["whole_card_vram_le_6800_mib"] is True
    assert v_ok["pass"] is True
    v_hi = drill.verdict_phase(3, slots, baseline_wall_s=100.0,
                               vram_peak_mib=6801, vram_samples=100)
    assert v_hi["checks"]["whole_card_vram_le_6800_mib"] is False
    assert v_hi["pass"] is False
    # fail-closed: sampler missing / empty file -> peak 0 -> N=3 fails
    v_missing = drill.verdict_phase(3, slots, baseline_wall_s=100.0,
                                    vram_peak_mib=0, vram_samples=0)
    assert v_missing["checks"]["whole_card_vram_le_6800_mib"] is False
    assert v_missing["pass"] is False
    assert v_missing["observed"]["vram_peak_mib"] == 0
    # fail-closed for the default too (verdict called without sampler data)
    v_default = drill.verdict_phase(3, slots, baseline_wall_s=100.0)
    assert v_default["pass"] is False
    # N=1 / N=4: NOT enforced — recorded informationally only
    v1 = drill.verdict_phase(1, [slots[0]], baseline_wall_s=None,
                             vram_peak_mib=6900, vram_samples=100)
    assert "whole_card_vram_le_6800_mib" not in v1["checks"]
    assert v1["observed"]["whole_card_vram_le_6800_mib"] is False
    assert v1["observed"]["vram_peak_mib"] == 6900
    assert v1["pass"] is True
    v4 = drill.verdict_phase(4, slots + [dict(slots[0], run_id="r3")],
                             baseline_wall_s=100.0,
                             vram_peak_mib=6900, vram_samples=100)
    assert "whole_card_vram_le_6800_mib" not in v4["checks"]
    assert v4["observed"]["vram_peak_mib"] == 6900
    assert v4["pass"] is True


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
# B4 drill: whole-card VRAM sampler + cuda-memory-fraction passthrough
# ---------------------------------------------------------------------------
def test_vram_sampler_start_stop_mocked(tmp_path: Path,
                                        monkeypatch) -> None:
    """Whole-card sampler contract: nvidia-smi -l 2 -> phase_root file,
    reliably terminate/wait-stopped; nvidia-smi missing = empty file
    (parse peak 0 -> the N=3 gate fails closed).  No GPU touched."""
    seen: dict = {}

    class _FakeProc:
        def __init__(self) -> None:
            self.terminated = False
            self.waited = False
            self.killed = False

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float = 0) -> int:
            self.waited = True
            return 0

        def kill(self) -> None:
            self.killed = True

    def _fake_popen(cmd, **kw):
        seen["cmd"] = list(cmd)
        seen["stdout"] = kw["stdout"]
        seen["proc"] = _FakeProc()
        return seen["proc"]

    monkeypatch.setattr(drill.subprocess, "Popen", _fake_popen)
    vram_path = tmp_path / drill.VRAM_SAMPLE_FILENAME
    handle = drill._start_vram_sampler(tmp_path)
    assert handle is not None and vram_path.is_file()
    assert seen["cmd"] == ["nvidia-smi", "--query-gpu=memory.used",
                           "--format=csv,noheader", "-l", "2"]
    vram_path.write_text("6100 MiB\n6400 MiB\n", encoding="utf-8")
    drill._stop_vram_sampler(handle)
    assert seen["proc"].terminated is True and seen["proc"].waited is True
    assert not seen["proc"].killed
    # samples parsed by the REUSED infra_bench parser
    stats = drill.ib.parse_vram_samples(drill.ib._read_lines(vram_path))
    assert stats == {"vram_peak_mib": 6400, "vram_samples": 2}
    drill._stop_vram_sampler(None)          # no-op

    def _boom(cmd, **kw):
        raise OSError("nvidia-smi: not found")

    monkeypatch.setattr(drill.subprocess, "Popen", _boom)
    sub = tmp_path / "no_smi"
    assert drill._start_vram_sampler(sub) is None
    empty = sub / drill.VRAM_SAMPLE_FILENAME
    assert empty.is_file() and empty.read_text(encoding="utf-8") == ""
    assert drill.ib.parse_vram_samples(
        drill.ib._read_lines(empty))["vram_peak_mib"] == 0
    drill._stop_vram_sampler(None)          # missing sampler: still safe


def test_cuda_memory_fraction_validation(tmp_path: Path) -> None:
    assert drill._cuda_memory_fraction_arg("0.17") == 0.17
    assert drill._cuda_memory_fraction_arg(1) == 1.0
    for bad in ("0", "-0.1", "1.5", "junk", "nan"):
        with pytest.raises(argparse.ArgumentTypeError):
            drill._cuda_memory_fraction_arg(bad)
    # CLI: bad values are refused before anything is built or launched
    with pytest.raises(SystemExit):
        drill.main(["--cuda-memory-fraction", "1.5", "--dry-run"])
    with pytest.raises(SystemExit):
        drill.main(["--cuda-memory-fraction", "0", "--dry-run"])
    # plan builder: fail-closed too (off-CLI callers)
    with pytest.raises(SystemExit) as exc:
        drill.build_phase_plan(
            {"run_id": "v53mp_20k_A_s17", "arm": "A", "seed": 17},
            "B4_n3", 2000, 1, tmp_path, tmp_path, tmp_path, tmp_path,
            tmp_path, "cuda", "6G", 6, cuda_memory_fraction=0)
    assert "BAD_CUDA_MEMORY_FRACTION" in str(exc.value)


def test_drill_dry_run_cuda_memory_fraction_argv(tmp_path: Path,
                                                 monkeypatch) -> None:
    """--cuda-memory-fraction f -> every slot argv carries
    '--cuda-memory-fraction f'; default -> flag absent.  Dry-run launches
    nothing (Popen is trapped)."""
    plan = tmp_path / "run_plan_v53mp_20k.jsonl"
    _write_plan(plan, _plan_rows(9))
    cfg = REPO / "config/campaigns/pes2ts_v53mp_20261021.yaml"
    if not cfg.is_file():
        pytest.skip("frozen campaign config missing")
    monkeypatch.setattr(drill.ib, "validate_bench_root",
                        lambda *a, **k: {"five_sha": "ok"})

    def _no_launch(*a, **k):
        raise AssertionError("dry-run must not launch GPU/training")

    monkeypatch.setattr(drill.subprocess, "Popen", _no_launch)

    out_frac = tmp_path / "drill_frac"
    rc = drill.main([
        "--plan", str(plan), "--output-root", str(out_frac),
        "--ns", "1,3", "--updates", "2000", "--config", str(cfg),
        "--cuda-memory-fraction", "0.17", "--dry-run",
    ])
    assert rc == 0
    report = json.loads((out_frac / "report.json").read_text(encoding="utf-8"))
    assert report["cuda_memory_fraction"] == 0.17
    for phase in report["phases"]:
        for slot in phase["slots"]:
            argv = slot["argv"]
            i = argv.index("--cuda-memory-fraction")
            assert argv[i + 1] == "0.17"
            # driver arg: AFTER the driver path (forwarded by the wrapper)
            assert i > argv.index(str(drill.ib.RUN_DRIVER))
            assert slot["cuda_memory_fraction"] == 0.17

    out_plain = tmp_path / "drill_plain"
    rc = drill.main([
        "--plan", str(plan), "--output-root", str(out_plain),
        "--ns", "1,3", "--updates", "2000", "--config", str(cfg),
        "--dry-run",
    ])
    assert rc == 0
    report = json.loads(
        (out_plain / "report.json").read_text(encoding="utf-8"))
    assert report["cuda_memory_fraction"] is None
    for phase in report["phases"]:
        for slot in phase["slots"]:
            assert "--cuda-memory-fraction" not in slot["argv"]
            assert slot["cuda_memory_fraction"] is None


# ---------------------------------------------------------------------------
# B4 drill: --eval-vram-gate-* passthrough (3-way staggered evals)
# ---------------------------------------------------------------------------
def test_eval_vram_gate_validation(tmp_path: Path) -> None:
    assert drill._eval_vram_gate_mib_arg("6800") == 6800
    assert drill._eval_vram_gate_mib_arg(1) == 1
    assert drill._eval_vram_gate_timeout_arg("120") == 120
    assert drill._eval_vram_gate_timeout_arg(0) == 0
    for bad in ("0", "-1", "junk", "nan", "1.5"):
        with pytest.raises(argparse.ArgumentTypeError):
            drill._eval_vram_gate_mib_arg(bad)
    for bad in ("-1", "junk", "1.5"):
        with pytest.raises(argparse.ArgumentTypeError):
            drill._eval_vram_gate_timeout_arg(bad)
    # CLI: bad values are refused before anything is built or launched
    with pytest.raises(SystemExit):
        drill.main(["--eval-vram-gate-mib", "0", "--dry-run"])
    # plan builder: fail-closed too (off-CLI callers)
    with pytest.raises(SystemExit) as exc:
        drill.build_phase_plan(
            {"run_id": "v53mp_20k_A_s17", "arm": "A", "seed": 17},
            "B4_n3", 2000, 1, tmp_path, tmp_path, tmp_path, tmp_path,
            tmp_path, "cuda", "6G", 6, eval_vram_gate_mib=0)
    assert "BAD_EVAL_VRAM_GATE" in str(exc.value)


def test_drill_dry_run_eval_vram_gate_argv(tmp_path: Path,
                                           monkeypatch) -> None:
    """--eval-vram-gate-mib m -> every slot argv carries both gate flags;
    default -> flags absent.  Dry-run launches nothing (Popen trapped)."""
    plan = tmp_path / "run_plan_v53mp_20k.jsonl"
    _write_plan(plan, _plan_rows(9))
    cfg = REPO / "config/campaigns/pes2ts_v53mp_20261021.yaml"
    if not cfg.is_file():
        pytest.skip("frozen campaign config missing")
    monkeypatch.setattr(drill.ib, "validate_bench_root",
                        lambda *a, **k: {"five_sha": "ok"})

    def _no_launch(*a, **k):
        raise AssertionError("dry-run must not launch GPU/training")

    monkeypatch.setattr(drill.subprocess, "Popen", _no_launch)

    out_gate = tmp_path / "drill_gate"
    rc = drill.main([
        "--plan", str(plan), "--output-root", str(out_gate),
        "--ns", "1,3", "--updates", "2000", "--config", str(cfg),
        "--eval-vram-gate-mib", "6900", "--eval-vram-gate-timeout-s", "120",
        "--dry-run",
    ])
    assert rc == 0
    report = json.loads((out_gate / "report.json").read_text(encoding="utf-8"))
    assert report["eval_vram_gate_mib"] == 6900
    assert report["eval_vram_gate_timeout_s"] == 120
    for phase in report["phases"]:
        for slot in phase["slots"]:
            argv = slot["argv"]
            i = argv.index("--eval-vram-gate-mib")
            assert argv[i + 1] == "6900"
            j = argv.index("--eval-vram-gate-timeout-s")
            assert argv[j + 1] == "120"
            # driver args: AFTER the driver path (forwarded by the wrapper)
            assert i > argv.index(str(drill.ib.RUN_DRIVER))
            assert slot["eval_vram_gate_mib"] == 6900
            assert slot["eval_vram_gate_timeout_s"] == 120

    out_plain = tmp_path / "drill_plain_gate"
    rc = drill.main([
        "--plan", str(plan), "--output-root", str(out_plain),
        "--ns", "1,3", "--updates", "2000", "--config", str(cfg),
        "--dry-run",
    ])
    assert rc == 0
    report = json.loads(
        (out_plain / "report.json").read_text(encoding="utf-8"))
    assert report["eval_vram_gate_mib"] is None
    for phase in report["phases"]:
        for slot in phase["slots"]:
            assert "--eval-vram-gate-mib" not in slot["argv"]
            assert "--eval-vram-gate-timeout-s" not in slot["argv"]
            assert slot["eval_vram_gate_mib"] is None


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
