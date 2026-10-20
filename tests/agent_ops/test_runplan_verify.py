"""run_plan builder/verifier tests (mission: test_runplan.py, lane W-LOSS).

Covers ``scripts/agent_ops/runplan/build_r3_runplan.py`` + ``verify.py``:
exact 66 R3 + 33 V5.2 planned lines (30h plan §4.2 + §5.5), 99 unique
run_ids, exact per-substage counts, resolvable reuse targets, the top-level
PD-3 ``pd3_note``, and fail-closed behaviour of every guard (tampered plans,
missing/corrupt plans, invalid shas, frozen-file overwrite refusal).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.agent_ops.ledger.run_ledger import compute_run_id
from scripts.agent_ops.runplan import build_r3_runplan as builder
from scripts.agent_ops.runplan import verify as verifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "PES2TS_30H_R3_V52_V53"


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_cli(module: str, args):
    return subprocess.run(
        [sys.executable, "-m", module] + args,
        cwd=str(PROJECT_ROOT),
        env=_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.fixture
def plan_path(tmp_path) -> Path:
    out = tmp_path / "run_plan.jsonl"
    builder.write_plan(out)
    return out


def read_plan(path: Path):
    header, runs = None, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if rec.get("record_type") == "plan_header":
            header = rec
        else:
            runs.append(rec)
    return header, runs


def rewrite_plan(path: Path, header, runs) -> None:
    lines = []
    if header is not None:
        lines.append(json.dumps(header, sort_keys=True))
    lines.extend(json.dumps(r, sort_keys=True) for r in runs)
    path.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# build: exact matrix
# ---------------------------------------------------------------------------


def test_build_summary_exact_66_plus_33(tmp_path):
    summary = builder.write_plan(tmp_path / "run_plan.jsonl")
    assert summary["R3"] == 66
    assert summary["V5.2"] == 33
    assert summary["runs"] == 99
    assert summary["unique_run_ids"] == 99
    assert summary["pd3_note"] == builder.PD3_NOTE


def test_header_carries_exact_pd3_note(plan_path):
    header, runs = read_plan(plan_path)
    assert header is not None
    assert header["pd3_note"] == (
        "extra-3 beyond 63 = F-bidir per 30h §4.2; matched forward-only control = "
        "F2-forward fold0×3 inside F-main (reuse, not new runs); flagged for human "
        "confirm at E0 sign-off"
    )
    assert header["budget"] == {"R3": 66, "V5.2": 33}
    assert header["seeds"] == [17, 29, 43]
    assert len(runs) == 99


def test_every_line_has_required_fields_and_status(plan_path):
    _, runs = read_plan(plan_path)
    for rec in runs:
        for key in (
            "run_id",
            "campaign_id",
            "phase",
            "substage",
            "arm",
            "fold",
            "seed",
            "variant",
            "reuses",
            "status",
            "decision_note",
            "config_sha256",
            "data_sha256",
            "teacher_sha256",
            "features_sha256",
            "split_sha256",
        ):
            assert key in rec, "missing {} in {}".format(key, rec.get("run_id"))
        assert rec["status"] == "planned"
        assert rec["campaign_id"] == CAMPAIGN
        assert isinstance(rec["reuses"], list)


def test_unique_run_ids_and_no_cross_arm_duplicates(plan_path):
    _, runs = read_plan(plan_path)
    ids = [r["run_id"] for r in runs]
    assert len(ids) == len(set(ids)) == 99
    keys = [(r["campaign_id"], r["phase"], r["arm"], r["fold"], r["seed"]) for r in runs]
    assert len(keys) == len(set(keys)) == 99
    # run_id is the hashed identity: 16 lowercase hex chars
    assert all(len(r) == 16 and all(c in "0123456789abcdef" for c in r) for r in ids)


def test_per_substage_counts_exact(plan_path):
    _, runs = read_plan(plan_path)
    counts = {}
    for rec in runs:
        counts[rec["substage"]] = counts.get(rec["substage"], 0) + 1
    assert counts == builder.SUBSTAGE_COUNTS
    phases = {}
    for rec in runs:
        phases[rec["phase"]] = phases.get(rec["phase"], 0) + 1
    assert phases == {"R3": 66, "V5.2": 33}


def test_matrix_membership_details(plan_path):
    _, runs = read_plan(plan_path)

    def select(substage):
        return [r for r in runs if r["substage"] == substage]

    fmain = select("F-main")
    assert {(r["arm"], r["fold"]) for r in fmain} == {
        (a, 0) for a in ("F0", "F1", "F2", "F3")
    }
    assert {r["seed"] for r in fmain} == {17, 29, 43}
    assert all(r["variant"] == "forward" for r in fmain)

    bidir = select("F-bidir")
    assert len(bidir) == 3
    assert all(r["arm"] == "F2-bidir" and r["fold"] == 0 and r["variant"] == "bidir" for r in bidir)

    ablate = select("F-ablate")
    assert len(ablate) == 3
    assert all(
        r["arm"] == "F2-intent-off" and r["fold"] == 0 and r["variant"] == "full_mask_PD6"
        for r in ablate
    )
    assert all("PD-6" in r["decision_note"] for r in ablate)

    b = select("B")
    assert len(b) == 12
    assert {r["fold"] for r in b} == {1, 2, 3, 4}  # fold0 is reuse, not a new run
    assert all("B-fold0→F-winner" in r["decision_note"] for r in b)

    l_arm = select("L")
    assert {r["arm"] for r in l_arm} == {"S2", "S10", "D2", "D10"}
    assert len(l_arm) == 12

    v = select("V")
    assert {r["arm"] for r in v} == {"V-top1", "V-top2"}
    assert {r["fold"] for r in v} == {1, 2}
    assert len(v) == 12

    r_runs = select("R")
    assert {r["arm"] for r in r_runs} == {"R1", "R2"}
    assert all(r["fold"] == 0 for r in r_runs)
    assert all("R-R0→main" in r["decision_note"] for r in r_runs)
    assert len(r_runs) == 6

    c = select("C")
    assert {r["fold"] for r in c} == {3, 4}
    assert all("unseal" in r["decision_note"] for r in c)
    assert len(c) == 6

    for arm, n in (("Q", 9), ("T", 9), ("QT", 9)):
        group = [r for r in runs if r["substage"] == arm]
        assert len(group) == n
        assert {r["fold"] for r in group} == {0, 1, 2}

    qt_confirm = select("QT-confirm")
    assert len(qt_confirm) == 6
    assert all(
        r["arm"] == "QT" and r["variant"] == "confirm" and r["fold"] in (3, 4)
        for r in qt_confirm
    )
    assert all("M0→R3 checkpoint" in r["decision_note"] for r in qt_confirm)


def test_reuses_targets_resolve(plan_path):
    _, runs = read_plan(plan_path)
    populated = [r for r in runs if r["reuses"]]
    assert populated, "expected machine-checkable reuse links in the plan"
    substage_names = {r["substage"] for r in runs}
    run_ids = {r["run_id"] for r in runs}
    for rec in populated:
        for target in rec["reuses"]:
            if target.startswith("substage:"):
                assert target[len("substage:"):] in substage_names
            elif target.startswith("phase:"):
                assert target[len("phase:"):] in {r["phase"] for r in runs}
            else:
                assert target in run_ids


# ---------------------------------------------------------------------------
# verify: pass + fail-closed refusals
# ---------------------------------------------------------------------------


def test_verify_passes_on_fresh_build(plan_path):
    ok, problems, summary = verifier.verify_plan(plan_path)
    assert ok, problems
    assert problems == []
    assert summary["R3"] == 66 and summary["V5.2"] == 33
    assert summary["total"] == 99 and summary["unique_run_ids"] == 99


def test_verify_cli_prints_66_33_line(plan_path):
    proc = _run_cli("scripts.agent_ops.runplan.verify", ["--plan", str(plan_path)])
    assert proc.returncode == 0, proc.stderr
    assert (
        "run_plan VERIFY PASS: R3=66 V5.2=33 total=99 unique_run_ids=99"
        in proc.stdout
    )


def test_verify_missing_plan_refused(tmp_path):
    ok, problems, _ = verifier.verify_plan(tmp_path / "nope.jsonl")
    assert not ok
    assert any(p.startswith("PLAN_MISSING") for p in problems)
    proc = _run_cli(
        "scripts.agent_ops.runplan.verify", ["--plan", str(tmp_path / "nope.jsonl")]
    )
    assert proc.returncode == 1


def test_verify_duplicate_line_refused(plan_path):
    header, runs = read_plan(plan_path)
    runs.append(dict(runs[0]))  # exact duplicate run_id + cross-arm key
    rewrite_plan(plan_path, header, runs)
    ok, problems, _ = verifier.verify_plan(plan_path)
    assert not ok
    assert any(p.startswith("RUN_ID: duplicate") for p in problems)
    assert any(p.startswith("CROSS_ARM_DUP") for p in problems)


def test_verify_illegal_seed_refused(plan_path):
    header, runs = read_plan(plan_path)
    runs[0]["seed"] = 99  # not in {17, 29, 43}
    rewrite_plan(plan_path, header, runs)
    ok, problems, _ = verifier.verify_plan(plan_path)
    assert not ok
    assert any(p.startswith("SEED") for p in problems)


def test_verify_broken_reuse_target_refused(plan_path):
    header, runs = read_plan(plan_path)
    target = next(r for r in runs if r["reuses"])
    target["reuses"] = ["substage:DoesNotExist"]
    rewrite_plan(plan_path, header, runs)
    ok, problems, _ = verifier.verify_plan(plan_path)
    assert not ok
    assert any(p.startswith("REUSE") and "DoesNotExist" in p for p in problems)


def test_verify_missing_pd3_note_refused(plan_path):
    header, runs = read_plan(plan_path)
    header["pd3_note"] = "silently dropped"
    rewrite_plan(plan_path, header, runs)
    ok, problems, _ = verifier.verify_plan(plan_path)
    assert not ok
    assert any(p.startswith("PD3_NOTE") for p in problems)


def test_verify_wrong_count_refused(plan_path):
    header, runs = read_plan(plan_path)
    runs.pop()  # drop one V5.2 line
    rewrite_plan(plan_path, header, runs)
    ok, problems, _ = verifier.verify_plan(plan_path)
    assert not ok
    assert any("COUNT" in p for p in problems) or any("SUBSTAGE" in p for p in problems)


def test_verify_run_id_identity_drift_refused(plan_path):
    header, runs = read_plan(plan_path)
    runs[0]["config_sha256"] = "f" * 64  # changed config, stale run_id
    rewrite_plan(plan_path, header, runs)
    ok, problems, _ = verifier.verify_plan(plan_path)
    assert not ok
    assert any("identity drift" in p for p in problems)


def test_verify_corrupt_json_refused(plan_path):
    plan_path.write_text('{"record_type": "plan_header", broken\n', encoding="utf-8")
    ok, problems, _ = verifier.verify_plan(plan_path)
    assert not ok
    assert any(p.startswith("PLAN_CORRUPT") for p in problems)


def test_verify_non_planned_status_refused(plan_path):
    header, runs = read_plan(plan_path)
    runs[0]["status"] = "done"  # planning artifact may never carry execution state
    rewrite_plan(plan_path, header, runs)
    ok, problems, _ = verifier.verify_plan(plan_path)
    assert not ok
    assert any("status" in p for p in problems)


# ---------------------------------------------------------------------------
# build guards (fail-closed)
# ---------------------------------------------------------------------------


def test_build_refuses_overwrite_unless_force(tmp_path):
    out = tmp_path / "run_plan.jsonl"
    builder.write_plan(out)
    with pytest.raises(builder.BuildError) as excinfo:
        builder.write_plan(out)
    assert "PLAN_EXISTS" in str(excinfo.value)
    again = builder.write_plan(out, force=True)  # explicit rebuild allowed
    assert again["unique_run_ids"] == 99


@pytest.mark.parametrize(
    "bad_shas",
    [
        {"config_sha256": "not-a-sha"},
        {"data_sha256": "AB" * 32},  # uppercase is not a valid digest here
        {"nope_sha256": "0" * 64},
    ],
)
def test_build_invalid_shas_refused(tmp_path, bad_shas):
    with pytest.raises(builder.BuildError) as excinfo:
        builder.write_plan(tmp_path / "p.jsonl", shas=bad_shas)
    assert "INVALID_SHAS" in str(excinfo.value)


def test_build_cli_roundtrip_and_frozen_refusal(tmp_path):
    out = tmp_path / "run_plan.jsonl"
    proc = _run_cli(
        "scripts.agent_ops.runplan.build_r3_runplan", ["--out", str(out)]
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["R3"] == 66 and payload["V5.2"] == 33

    proc = _run_cli("scripts.agent_ops.runplan.verify", ["--plan", str(out)])
    assert proc.returncode == 0, proc.stderr
    assert "R3=66 V5.2=33 total=99 unique_run_ids=99" in proc.stdout

    # frozen artifact: rebuild without --force is refused (exit 2)
    proc = _run_cli(
        "scripts.agent_ops.runplan.build_r3_runplan", ["--out", str(out)]
    )
    assert proc.returncode == 2
    assert "PLAN_EXISTS" in proc.stderr


def test_run_id_matches_ledger_hash_scheme(plan_path):
    """Plan run_ids are the ledger's hashed scheme (changed config -> new id)."""
    _, runs = read_plan(plan_path)
    rec = dict(runs[0])
    assert compute_run_id(rec, scheme="hashed") == rec["run_id"]
    rec2 = dict(rec)
    rec2["config_sha256"] = "e" * 64
    assert compute_run_id(rec2, scheme="hashed") != rec["run_id"]
