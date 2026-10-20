"""Allowlist + frozen-file denylist tests (mission: test_allowlist.py, lane W-LOSS).

Covers ``scripts/agent_ops/supervisor/allowlist.py``: generation from
``parallel_config.json`` write_scopes + E0 additions, sha256 recording of the
three frozen files, ``check_merge`` write-scope enforcement against a real
git diff (exit-1 semantics), ``check_frozen`` drift alarms, and fail-closed
refusals for missing/corrupt policy inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.agent_ops.supervisor import allowlist as al

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REAL_CONFIG = PROJECT_ROOT / al.DEFAULT_CONFIG_PATH


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


def _run_cli(args, cwd=PROJECT_ROOT):
    return subprocess.run(
        [sys.executable, "-m", "scripts.agent_ops.supervisor.allowlist"] + args,
        cwd=str(cwd),
        env=_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo)] + list(args),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fake_root(tmp_path) -> Path:
    """Minimal repo root holding the real config copies + the 3 frozen files."""
    root = tmp_path / "repo"
    config = root / al.DEFAULT_CONFIG_PATH
    config.parent.mkdir(parents=True)
    shutil.copyfile(REAL_CONFIG, config)
    for rel in al.FROZEN_FILES:
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PROJECT_ROOT / rel, dst)
    (root / "scripts" / "agent_ops" / "ledger").mkdir(parents=True)
    (root / "scripts" / "agent_ops" / "__init__.py").write_text("", encoding="utf-8")
    (root / "scripts" / "agent_ops" / "ledger" / "a.py").write_text("X = 1\n", encoding="utf-8")
    return root


def _git_init(repo: Path, branch: str = "base") -> None:
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "wp-l2@test")
    _git(repo, "config", "user.name", "wp-l2")
    _git(repo, "checkout", "-q", "-b", branch)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


def test_generate_records_write_scopes_and_e0_additions(tmp_path):
    payload = al.generate_allowlist(tmp_path / "allow.json", repo_root=PROJECT_ROOT)
    patterns = payload["write_allowlist"]
    # lane write_scopes from parallel_config
    assert "scripts/agent_ops/ledger/**" in patterns
    assert "scripts/agent_ops/gpu_queue/**" in patterns
    assert "tests/agent_ops/test_ledger_*.py" in patterns
    # E0 additions
    for addition in al.E0_ADDITIONS:
        assert addition in patterns
    # shared supervisor-owned files
    assert "scripts/agent_ops/__init__.py" in payload["shared_files"]
    assert "tests/agent_ops/__init__.py" in payload["shared_files"]
    assert any("ground_truth" in p for p in payload["never_write_all_lanes"])


def test_generate_records_frozen_sha256_prefixes(tmp_path):
    payload = al.generate_allowlist(tmp_path / "allow.json", repo_root=PROJECT_ROOT)
    frozen = payload["frozen_files"]
    assert sorted(frozen.keys()) == sorted(al.FROZEN_FILES)
    for rel in al.FROZEN_FILES:
        expected = sha256_of(PROJECT_ROOT / rel)
        assert frozen[rel]["sha256"] == expected
        assert frozen[rel]["prefix"] == expected[:16]


def test_generate_writes_loadable_json(tmp_path):
    out = tmp_path / "sub" / "allow.json"
    al.generate_allowlist(out, repo_root=PROJECT_ROOT)
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["kind"] == "e0_write_allowlist"
    assert isinstance(data["write_allowlist"], list) and data["write_allowlist"]


# ---------------------------------------------------------------------------
# generate: fail-closed refusals
# ---------------------------------------------------------------------------


def test_generate_missing_config_refused(tmp_path):
    with pytest.raises(al.AllowlistError) as excinfo:
        al.generate_allowlist(tmp_path / "a.json", repo_root=tmp_path / "empty")
    assert "CONFIG_MISSING" in str(excinfo.value)


def test_generate_corrupt_config_refused(tmp_path):
    config = tmp_path / "config.json"
    config.write_text("{not json", encoding="utf-8")
    with pytest.raises(al.AllowlistError) as excinfo:
        al.generate_allowlist(tmp_path / "a.json", config_path=config, repo_root=PROJECT_ROOT)
    assert "CONFIG_CORRUPT" in str(excinfo.value)

    config.write_text(json.dumps({"workflows": {}}), encoding="utf-8")
    with pytest.raises(al.AllowlistError) as excinfo:
        al.generate_allowlist(tmp_path / "a.json", config_path=config, repo_root=PROJECT_ROOT)
    assert "CONFIG_CORRUPT" in str(excinfo.value)


def test_generate_missing_frozen_file_refused(tmp_path):
    config = tmp_path / "config.json"
    shutil.copyfile(REAL_CONFIG, config)
    with pytest.raises(al.AllowlistError) as excinfo:
        al.generate_allowlist(
            tmp_path / "a.json", config_path=config, repo_root=tmp_path / "nofiles"
        )
    assert "FROZEN_FILE_MISSING" in str(excinfo.value)


def test_generate_missing_workflow_scope_refused(tmp_path):
    config = json.loads(REAL_CONFIG.read_text(encoding="utf-8"))
    config["workflows"]["W-LOSS"]["write_scopes"] = []
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(al.AllowlistError) as excinfo:
        al.generate_allowlist(tmp_path / "a.json", config_path=config_path, repo_root=PROJECT_ROOT)
    assert "CONFIG_CORRUPT" in str(excinfo.value)


# ---------------------------------------------------------------------------
# check_merge against a real git diff
# ---------------------------------------------------------------------------


def test_check_merge_flags_violation_and_allows_scope(fake_root):
    _git_init(fake_root, branch="base")
    _git(fake_root, "checkout", "-q", "-b", "lane")

    # in-scope change: scripts/agent_ops/ledger/**
    (fake_root / "scripts" / "agent_ops" / "ledger" / "a.py").write_text(
        "X = 2\n", encoding="utf-8"
    )
    # shared supervisor-owned file (allowed by shared_files)
    (fake_root / "scripts" / "agent_ops" / "__init__.py").write_text(
        "# supervisor owns\n", encoding="utf-8"
    )
    # out-of-scope change: business code outside E0 additions
    forbidden = fake_root / "pes2ts_core" / "pesn" / "train.py"
    forbidden.parent.mkdir(parents=True, exist_ok=True)
    forbidden.write_text("print('nope')\n", encoding="utf-8")
    _git(fake_root, "add", "-A")
    _git(fake_root, "commit", "-q", "-m", "lane changes")

    payload = al.generate_allowlist(fake_root / "allow.json", repo_root=fake_root)

    result = al.check_merge("lane", "base", repo_root=fake_root, allowlist=payload)
    assert result["ok"] is False
    assert result["violations"] == ["pes2ts_core/pesn/train.py"]
    assert "scripts/agent_ops/ledger/a.py" in result["files"]
    assert "scripts/agent_ops/__init__.py" in result["files"]  # shared: not a violation


def test_check_merge_clean_branch_passes(fake_root):
    _git_init(fake_root, branch="base")
    _git(fake_root, "checkout", "-q", "-b", "lane-clean")
    (fake_root / "scripts" / "agent_ops" / "ledger" / "a.py").write_text(
        "X = 3\n", encoding="utf-8"
    )
    _git(fake_root, "add", "-A")
    _git(fake_root, "commit", "-q", "-m", "in scope")

    payload = al.generate_allowlist(fake_root / "allow.json", repo_root=fake_root)
    result = al.check_merge("lane-clean", "base", repo_root=fake_root, allowlist=payload)
    assert result["ok"] is True, result
    assert result["violations"] == []


def test_check_merge_never_write_all_lanes_is_hard_deny(fake_root):
    _git_init(fake_root, branch="base")
    _git(fake_root, "checkout", "-q", "-b", "lane-bad")
    out = fake_root / "outputs" / "pesn" / "v5_1" / "sneaky.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("x", encoding="utf-8")
    _git(fake_root, "add", "-A")
    _git(fake_root, "commit", "-q", "-m", "touch sealed")

    payload = al.generate_allowlist(fake_root / "allow.json", repo_root=fake_root)
    result = al.check_merge("lane-bad", "base", repo_root=fake_root, allowlist=payload)
    assert result["ok"] is False
    assert any("v5_1" in v for v in result["violations"])


def test_check_merge_missing_allowlist_refused(tmp_path, fake_root):
    _git_init(fake_root, branch="base")
    with pytest.raises(al.AllowlistError) as excinfo:
        al.check_merge("base", "base", repo_root=fake_root)  # no allowlist file
    assert "ALLOWLIST_MISSING" in str(excinfo.value)


def test_check_merge_corrupt_allowlist_refused(tmp_path, fake_root):
    _git_init(fake_root, branch="base")
    path = tmp_path / "allow.json"
    path.write_text("{bad", encoding="utf-8")
    with pytest.raises(al.AllowlistError) as excinfo:
        al.check_merge("base", "base", repo_root=fake_root, allowlist_path=path)
    assert "ALLOWLIST_CORRUPT" in str(excinfo.value)


def test_check_merge_unknown_revision_refused(fake_root):
    _git_init(fake_root, branch="base")
    payload = al.generate_allowlist(fake_root / "allow.json", repo_root=fake_root)
    with pytest.raises(al.AllowlistError) as excinfo:
        al.check_merge("base", "no-such-rev", repo_root=fake_root, allowlist=payload)
    assert "GIT_DIFF_FAILED" in str(excinfo.value)


# ---------------------------------------------------------------------------
# frozen-file drift alarm
# ---------------------------------------------------------------------------


def test_check_frozen_clean_then_tamper_alarm(fake_root):
    allow = fake_root / "allow.json"
    al.generate_allowlist(allow, repo_root=fake_root)

    clean = al.check_frozen(allowlist_path=allow, repo_root=fake_root)
    assert clean["ok"] is True, clean
    assert sorted(clean["frozen_files"]) == sorted(al.FROZEN_FILES)

    # tamper with one frozen file -> frozen-file-modified alarm
    victim = fake_root / al.FROZEN_FILES[0]
    victim.write_text(victim.read_text(encoding="utf-8") + "\n# drifted\n", encoding="utf-8")
    dirty = al.check_frozen(allowlist_path=allow, repo_root=fake_root)
    assert dirty["ok"] is False
    assert dirty["mismatches"][0]["file"] == al.FROZEN_FILES[0]
    assert dirty["mismatches"][0]["actual"] != dirty["mismatches"][0]["expected"]

    # deleted frozen file -> missing alarm too
    victim.unlink()
    gone = al.check_frozen(allowlist_path=allow, repo_root=fake_root)
    assert gone["ok"] is False
    assert gone["missing"] == [al.FROZEN_FILES[0]]


def test_check_frozen_missing_allowlist_refused(tmp_path):
    with pytest.raises(al.AllowlistError) as excinfo:
        al.check_frozen(allowlist_path=tmp_path / "none.json", repo_root=PROJECT_ROOT)
    assert "ALLOWLIST_MISSING" in str(excinfo.value)


# ---------------------------------------------------------------------------
# CLI exit codes (mission: violations listed, exit 1)
# ---------------------------------------------------------------------------


def test_cli_generate_check_merge_exit_codes(fake_root, tmp_path):
    allow = tmp_path / "allow.json"
    proc = _run_cli(["--root", str(fake_root), "generate", "--out", str(allow)])
    assert proc.returncode == 0, proc.stderr

    # clean merge -> exit 0
    _git_init(fake_root, branch="base")
    proc = _run_cli(
        [
            "--root",
            str(fake_root),
            "--allowlist",
            str(allow),
            "check-merge",
            "--base",
            "base",
            "--branch",
            "base",
        ]
    )
    assert proc.returncode == 0, proc.stderr
    assert "MERGE OK" in proc.stdout

    # violating merge -> exit 1 + listed violation
    _git(fake_root, "checkout", "-q", "-b", "lane")
    bad = fake_root / "pes2ts_core" / "pesn" / "oops.py"
    bad.write_text("x\n", encoding="utf-8")
    _git(fake_root, "add", "-A")
    _git(fake_root, "commit", "-q", "-m", "violation")
    proc = _run_cli(
        [
            "--root",
            str(fake_root),
            "--allowlist",
            str(allow),
            "check-merge",
            "--base",
            "base",
            "--branch",
            "lane",
        ]
    )
    assert proc.returncode == 1
    assert "WRITE_SCOPE_VIOLATION: pes2ts_core/pesn/oops.py" in proc.stderr
    assert "MERGE REFUSED" in proc.stderr

    # frozen check on the untouched frozen files -> exit 0
    proc = _run_cli(
        ["--root", str(fake_root), "--allowlist", str(allow), "check-frozen"]
    )
    assert proc.returncode == 0, proc.stderr
    assert "FROZEN OK" in proc.stdout


def test_cli_missing_config_exits_nonzero(tmp_path):
    proc = _run_cli(
        ["--root", str(tmp_path), "generate", "--out", str(tmp_path / "a.json")]
    )
    assert proc.returncode == 1
    assert "CONFIG_MISSING" in proc.stderr
