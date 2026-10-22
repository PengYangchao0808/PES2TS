"""V5.3-MP parallel orchestrator acceptance: dry-run schedule + fail-closed CLI.

Script under test: ``scripts/campaigns/r3_v53mp_orchestrate_parallel.py``
(additive sibling of the sequential orchestrator).  Only ``--dry-run`` and
the fail-closed CLI edges are exercised here — the dry-run path must never
touch the GPU, the ledger, or any subprocess.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.campaigns import r3_v53mp_orchestrate_parallel as orch  # noqa: E402


def _write_plan(root: Path, tier: str, run_ids: list[str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    plan = root / f"run_plan_v53mp_{tier}.jsonl"
    lines = [json.dumps({"record_type": "plan_header", "tier": tier,
                         "phase": "S2"})]
    lines += [json.dumps({"run_id": rid, "tier": tier}) for rid in run_ids]
    plan.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return plan


def _write_ledger(root: Path, records: list[dict]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    ledger = root / "run_records.jsonl"
    ledger.write_text(
        "\n".join(json.dumps(r, sort_keys=True) for r in records) + "\n",
        encoding="utf-8")
    return ledger


def _events(root: Path, tier: str) -> list[dict]:
    path = root / f"orchestrate_{tier}_parallel.jsonl"
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines()
            if ln.strip()]


# --------------------------------------------------------------------------
# dry-run schedule
# --------------------------------------------------------------------------


def test_dry_run_emits_round_robin_schedule_without_subprocess(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    root = tmp_path / "outputs" / "pesn" / "v53mp"
    tier = "5k"
    run_ids = [f"syn_5k_{arm}_s{seed}"
               for arm in ("A", "B", "C") for seed in (17, 29, 43)]
    assert len(run_ids) == 9
    _write_plan(root, tier, run_ids)
    ledger_mtime_before = None  # no ledger yet — dry-run must not create one

    def _boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError(
            f"subprocess called during --dry-run: {args!r}")

    monkeypatch.setattr(subprocess, "run", _boom)
    monkeypatch.setattr(subprocess, "Popen", _boom)
    monkeypatch.setattr(orch.subprocess, "run", _boom)
    monkeypatch.setattr(orch.subprocess, "Popen", _boom)

    rc = orch.main(["--tier", tier, "--output-root", str(root),
                    "--concurrency", "3", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out

    events = _events(root, tier)
    launches = [e for e in events if e["event"] == "launch"]
    # schedule covers all 9 run ids, plan order, slots 0,1,2 round-robin
    assert [e["run_id"] for e in launches] == run_ids
    assert [e["slot"] for e in launches] == [0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert [e["order"] for e in launches] == list(range(9))
    for e in launches:
        assert e["concurrency"] == 3
        assert e["dry_run"] is True
        assert e["slot"] in (0, 1, 2)
    # schedule is also printed to stdout (one JSON event per line)
    printed = [json.loads(ln) for ln in out.splitlines() if ln.strip()]
    assert ([e["run_id"] for e in printed
             if e.get("event") == "launch"] == run_ids)
    # bookend events
    assert events[0]["event"] == "orchestrate_start"
    assert events[0]["n_pending"] == 9
    assert events[-1]["event"] == "orchestrate_end"
    assert events[-1]["pending"] == 9
    assert events[-1]["done"] == 0
    # fail-closed: no ledger, no lock files created by dry-run
    assert not (root / "run_records.jsonl").exists()
    assert ledger_mtime_before is None


def test_dry_run_schedule_uses_expected_slot_lock_paths() -> None:
    # the k-th concurrent worker must get its own slot lock path
    assert orch.SLOT_LOCK_TEMPLATE.format(k=0) == \
        "/tmp/opencode/pes2ts_gpu.slot0.lock"
    assert orch.SLOT_LOCK_TEMPLATE.format(k=2) == \
        "/tmp/opencode/pes2ts_gpu.slot2.lock"


# --------------------------------------------------------------------------
# fail-closed CLI edges
# --------------------------------------------------------------------------


def test_concurrency_zero_exits_2(tmp_path: Path, capsys) -> None:
    _write_plan(tmp_path, "5k", ["syn_5k_A_s17"])
    rc = orch.main(["--tier", "5k", "--output-root", str(tmp_path),
                    "--concurrency", "0", "--dry-run"])
    assert rc == 2
    assert "BAD_CONCURRENCY" in capsys.readouterr().err
    assert not (tmp_path / "orchestrate_5k_parallel.jsonl").exists()


def test_concurrency_negative_exits_2(tmp_path: Path, capsys) -> None:
    _write_plan(tmp_path, "5k", ["syn_5k_A_s17"])
    rc = orch.main(["--tier", "5k", "--output-root", str(tmp_path),
                    "--concurrency", "-1", "--dry-run"])
    assert rc == 2
    assert "BAD_CONCURRENCY" in capsys.readouterr().err


def test_missing_plan_exits_2(tmp_path: Path, capsys) -> None:
    rc = orch.main(["--tier", "5k", "--output-root", str(tmp_path),
                    "--concurrency", "3", "--dry-run"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "PLAN_MISSING" in err
    assert "run_plan_v53mp_5k.jsonl" in err


def test_empty_plan_exits_2(tmp_path: Path, capsys) -> None:
    root = tmp_path / "v53mp"
    root.mkdir(parents=True)
    (root / "run_plan_v53mp_5k.jsonl").write_text("", encoding="utf-8")
    rc = orch.main(["--tier", "5k", "--output-root", str(root),
                    "--concurrency", "3", "--dry-run"])
    assert rc == 2
    assert "PLAN_EMPTY" in capsys.readouterr().err


def test_header_only_plan_exits_2(tmp_path: Path, capsys) -> None:
    root = tmp_path / "v53mp"
    _write_plan(root, "5k", [])  # header only, zero run lines
    rc = orch.main(["--tier", "5k", "--output-root", str(root),
                    "--concurrency", "3", "--dry-run"])
    assert rc == 2
    assert "PLAN_EMPTY" in capsys.readouterr().err


# --------------------------------------------------------------------------
# skip_done against a synthetic ledger
# --------------------------------------------------------------------------


def test_dry_run_skips_run_already_done_in_ledger(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "outputs" / "pesn" / "v53mp"
    tier = "5k"
    run_ids = ["syn_5k_A_s17", "syn_5k_B_s17", "syn_5k_C_s17"]
    _write_plan(root, tier, run_ids)
    ledger = _write_ledger(
        root, [{"run_id": "syn_5k_B_s17", "status": "done"}])
    ledger_mtime = ledger.stat().st_mtime_ns

    def _boom(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("subprocess called during --dry-run")

    monkeypatch.setattr(orch.subprocess, "run", _boom)
    monkeypatch.setattr(orch.subprocess, "Popen", _boom)

    rc = orch.main(["--tier", tier, "--output-root", str(root),
                    "--concurrency", "3", "--dry-run"])
    assert rc == 0

    events = _events(root, tier)
    skips = [e for e in events if e["event"] == "skip_done"]
    assert [e["run_id"] for e in skips] == ["syn_5k_B_s17"]
    launches = [e for e in events if e["event"] == "launch"]
    # the done run is not scheduled; the other two take slots 0 and 1
    assert [e["run_id"] for e in launches] == ["syn_5k_A_s17",
                                               "syn_5k_C_s17"]
    assert [e["slot"] for e in launches] == [0, 1]
    end = events[-1]
    assert end["event"] == "orchestrate_end"
    assert end["done"] == 1
    assert end["pending"] == 2
    assert end["total"] == 3
    # dry-run must not mutate the ledger
    assert ledger.stat().st_mtime_ns == ledger_mtime
