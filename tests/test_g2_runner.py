"""Tests for the xTB PATH subprocess runner (plan task 5).

Every subprocess scenario is driven by a self-contained fake ``xtb`` shell
script generated in ``tmp_path`` and injected via ``g2.xtb.executable``.
The real xTB binary is never executed by the default suite.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import pytest

from pes2ts_core.g2.runner import (
    PATH_INP_FILENAME,
    XtbNotFoundError,
    XTB_LOG_FILENAME,
    resolve_executable,
    run_xtb_path,
    write_path_inp,
)
from pes2ts_core.utils.hashing import sha256_file

#: The exact ``$path`` block produced from the plan-default parameters.
DEFAULT_PATH_INP_LINES: list[str] = [
    "$path",
    "   nrun=1",
    "   npoint=25",
    "   anopt=10",
    "   kpush=0.003",
    "   kpull=-0.015",
    "   ppull=0.05",
    "   alp=1.2",
    "$end",
]

_SUCCESS_SCRIPT = """\
#!/bin/sh
if [ "$1" = "--help" ]; then
  echo "Usage: fake xtb [options] <geometry>"
  exit 0
fi
echo "   * xtb version 6.7.1 (fake-success)"
echo "fake-path-stdout-marker"
echo "OMP=$OMP_NUM_THREADS"
echo "args: $@"
cat > xtbpath.xyz <<'EOF'
2
 energy: 0.000000 xtb: 6.7.1 (fake)
C 0.0 0.0 0.0
H 1.0 0.0 0.0
2
 energy: -1.000000 xtb: 6.7.1 (fake)
C 0.0 0.0 0.0
H 1.2 0.0 0.0
EOF
exit 0
"""

_CRASHED_SCRIPT = """\
#!/bin/sh
if [ "$1" = "--help" ]; then
  echo "Usage: fake xtb [options] <geometry>"
  exit 0
fi
echo "   * xtb version 6.7.1 (fake-crash)"
echo "crash-stdout-marker"
echo "crash-stderr-marker" >&2
exit 3
"""

_TIMEOUT_SCRIPT = """\
#!/bin/sh
if [ "$1" = "--help" ]; then
  echo "Usage: fake xtb [options] <geometry>"
  exit 0
fi
echo "   * xtb version 6.7.1 (fake-timeout)"
echo $$ > fake_parent.pid
sleep 300 &
echo $! > fake_child.pid
sleep 60
"""

_SEED_HELP_SCRIPT = """\
#!/bin/sh
if [ "$1" = "--help" ]; then
  printf 'Options:\\n  --seed INT\\n          set the random seed\\n'
  exit 0
fi
echo "   * xtb version 6.7.1 (fake-seed)"
echo "args: $@"
exit 0
"""

_EMPTY_HELP_SCRIPT = """\
#!/bin/sh
exit 0
"""


def _write_script(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    path.chmod(0o755)
    return path


def _config(executable: str | Path | None, **xtb_overrides: Any) -> dict[str, Any]:
    xtb: dict[str, Any] = {
        "executable": None if executable is None else str(executable),
        "fallback_paths": ["/nonexistent/xtb-a", "/nonexistent/xtb-b"],
        "threads": 2,
        "timeout_seconds": 10,
        "seed": 42,
    }
    xtb.update(xtb_overrides)
    return {"g2": {"xtb": xtb, "path": {}}}


def _write_geometry(directory: Path, name: str, marker: str) -> None:
    (directory / name).write_text(
        f"2\n{marker}\nC 0.0 0.0 0.0\nH 1.0 0.0 0.0\n", encoding="utf-8"
    )


def _wait_pid_gone(pid: int, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.05)
    return False


# ---------------------------------------------------------------------------
# resolve_executable
# ---------------------------------------------------------------------------


def test_resolve_executable_prefers_configured_executable(tmp_path: Path) -> None:
    exe = _write_script(tmp_path / "my-xtb", "#!/bin/sh\nexit 0\n")
    assert resolve_executable(_config(exe)) == exe.resolve()


def test_resolve_executable_configured_but_missing_raises(tmp_path: Path) -> None:
    missing = tmp_path / "nowhere" / "xtb"
    with pytest.raises(XtbNotFoundError, match="nowhere"):
        resolve_executable(_config(missing))


def test_resolve_executable_falls_back_to_path_lookup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = _write_script(bin_dir / "xtb", "#!/bin/sh\nexit 0\n")
    monkeypatch.setenv("PATH", str(bin_dir))
    assert resolve_executable(_config(None)) == exe.resolve()


def test_resolve_executable_uses_fallback_paths_in_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_dir))
    fallback = _write_script(tmp_path / "fallback-xtb", "#!/bin/sh\nexit 0\n")
    config = _config(
        None,
        fallback_paths=[str(tmp_path / "missing-xtb"), str(fallback)],
    )
    assert resolve_executable(config) == fallback.resolve()


def test_resolve_executable_raises_actionable_error_when_nothing_resolves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    monkeypatch.setenv("PATH", str(empty_dir))
    with pytest.raises(XtbNotFoundError) as excinfo:
        resolve_executable(_config(None))
    message = str(excinfo.value)
    assert "g2.xtb.executable" in message
    assert "/nonexistent/xtb-a" in message
    assert "/nonexistent/xtb-b" in message
    assert "install" in message.lower()


# ---------------------------------------------------------------------------
# write_path_inp
# ---------------------------------------------------------------------------


def test_write_path_inp_matches_contract_line_by_line(tmp_path: Path) -> None:
    target = tmp_path / PATH_INP_FILENAME
    write_path_inp(target, _config(None))
    assert target.read_text(encoding="utf-8").splitlines() == DEFAULT_PATH_INP_LINES


def test_write_path_inp_honors_config_overrides(tmp_path: Path) -> None:
    target = tmp_path / PATH_INP_FILENAME
    config = {
        "g2": {
            "path": {
                "nrun": 2,
                "npoint": 40,
                "anopt": 5,
                "kpush": 0.01,
                "kpull": -0.02,
                "ppull": 0.1,
                "alp": 1.5,
            }
        }
    }
    write_path_inp(target, config)
    assert target.read_text(encoding="utf-8").splitlines() == [
        "$path",
        "   nrun=2",
        "   npoint=40",
        "   anopt=5",
        "   kpush=0.01",
        "   kpull=-0.02",
        "   ppull=0.1",
        "   alp=1.5",
        "$end",
    ]


# ---------------------------------------------------------------------------
# run_xtb_path — fake-binary scenarios
# ---------------------------------------------------------------------------


def test_run_xtb_path_success_contract(tmp_path: Path) -> None:
    # Given: a fake xtb that succeeds and a prepared run directory
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fake = _write_script(tmp_path / "fake-success", _SUCCESS_SCRIPT)
    _write_geometry(run_dir, "start.xyz", "start")
    _write_geometry(run_dir, "end.xyz", "end")

    # When: the runner drives the fake binary
    result = run_xtb_path(run_dir, _config(fake), charge=0, uhf=0)

    # Then: the process outcome is recorded exactly
    assert result.returncode == 0
    assert result.timed_out is False
    assert result.duration_seconds > 0.0

    # And: the merged log exists in the run directory and holds the stdout
    log_path = run_dir / XTB_LOG_FILENAME
    assert result.log_path == str(log_path)
    log_text = log_path.read_text(encoding="utf-8")
    assert "fake-path-stdout-marker" in log_text

    # And: the version line and executable fingerprint are recorded
    assert result.xtb_version_line == "* xtb version 6.7.1 (fake-success)"
    assert result.executable == str(fake.resolve())
    assert result.executable_sha256 == sha256_file(fake)

    # And: argv matches the frozen contract item-by-item (no seed, no extra)
    assert result.argv == (
        str(fake.resolve()),
        "start.xyz",
        "--path",
        "end.xyz",
        "--input",
        "path.inp",
        "-P",
        "2",
        "--gfn",
        "2",
        "--chrg",
        "0",
        "--uhf",
        "0",
    )
    # And: the delivered arguments and environment reach the process, whose
    # cwd is the run directory (the fake wrote xtbpath.xyz there)
    assert " ".join(result.argv[1:]) in log_text
    assert "OMP=2" in log_text
    assert (run_dir / "xtbpath.xyz").is_file()

    # And: the runner itself wrote the contract path.inp into the run dir
    assert (run_dir / PATH_INP_FILENAME).read_text(
        encoding="utf-8"
    ).splitlines() == DEFAULT_PATH_INP_LINES

    # And: the environment is the full environment plus OMP_NUM_THREADS
    assert result.env == {**os.environ, "OMP_NUM_THREADS": "2"}

    # And: the success fake's help advertises no seed flag
    assert result.seed_supported is False
    assert result.seed is None


def test_run_xtb_path_appends_extra_args(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fake = _write_script(tmp_path / "fake-success", _SUCCESS_SCRIPT)

    result = run_xtb_path(run_dir, _config(fake), extra_args=("--strict",))

    assert result.argv[-1] == "--strict"
    assert result.argv[:-1] == (
        str(fake.resolve()),
        "start.xyz",
        "--path",
        "end.xyz",
        "--input",
        "path.inp",
        "-P",
        "2",
        "--gfn",
        "2",
        "--chrg",
        "0",
        "--uhf",
        "0",
    )
    log_text = (run_dir / XTB_LOG_FILENAME).read_text(encoding="utf-8")
    assert (
        "args: start.xyz --path end.xyz --input path.inp -P 2 --gfn 2 "
        "--chrg 0 --uhf 0 --strict"
    ) in log_text


def test_run_xtb_path_crashed_exit_code_surfaces(tmp_path: Path) -> None:
    # Given: a fake xtb that exits 3 with stdout and stderr output
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fake = _write_script(tmp_path / "fake-crash", _CRASHED_SCRIPT)

    # When: the runner drives it, no exception is swallowed
    result = run_xtb_path(run_dir, _config(fake), charge=0, uhf=0)

    # Then: the exit code surfaces in the result
    assert result.returncode == 3
    assert result.timed_out is False
    # And: stdout and stderr are merged into the log
    log_text = (run_dir / XTB_LOG_FILENAME).read_text(encoding="utf-8")
    assert "crash-stdout-marker" in log_text
    assert "crash-stderr-marker" in log_text


def test_run_xtb_path_timeout_kills_whole_group_and_keeps_partial_log(
    tmp_path: Path,
) -> None:
    # Given: a fake xtb that spawns a background child and sleeps far longer
    # than the configured 2-second timeout
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fake = _write_script(tmp_path / "fake-timeout", _TIMEOUT_SCRIPT)

    # When: the runner times out
    result = run_xtb_path(run_dir, _config(fake, timeout_seconds=2), charge=0, uhf=0)

    # Then: the timeout is recorded and the whole group was killed
    assert result.timed_out is True
    assert result.returncode is not None
    assert result.returncode != 0
    assert result.duration_seconds > 1.0
    # And: the partial log (written before the kill) is captured
    assert result.xtb_version_line == "* xtb version 6.7.1 (fake-timeout)"
    # And: neither the fake binary nor its background child survives
    parent_pid = int((run_dir / "fake_parent.pid").read_text())
    child_pid = int((run_dir / "fake_child.pid").read_text())
    assert _wait_pid_gone(parent_pid), "fake xtb process survived the timeout"
    assert _wait_pid_gone(child_pid), "background child survived the group kill"


def test_run_xtb_path_passes_seed_when_help_advertises_it(tmp_path: Path) -> None:
    # Given: a fake whose --help advertises --seed and a configured seed of 7
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fake = _write_script(tmp_path / "fake-seed", _SEED_HELP_SCRIPT)

    # When
    result = run_xtb_path(run_dir, _config(fake, seed=7), charge=0, uhf=0)

    # Then: the advertised flag carries the configured seed
    assert result.seed_supported is True
    assert result.seed == 7
    assert result.argv[-2:] == ("--seed", "7")
    log_text = (run_dir / XTB_LOG_FILENAME).read_text(encoding="utf-8")
    assert "--seed 7" in log_text


def test_run_xtb_path_default_seed_is_42_when_supported(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fake = _write_script(tmp_path / "fake-seed", _SEED_HELP_SCRIPT)

    result = run_xtb_path(run_dir, _config(fake), charge=0, uhf=0)

    assert result.seed_supported is True
    assert result.seed == 42
    assert result.argv[-2:] == ("--seed", "42")


def test_run_xtb_path_empty_help_means_seed_unsupported(tmp_path: Path) -> None:
    # Given: a fake binary that prints nothing at all (unparseable help)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    fake = _write_script(tmp_path / "fake-empty", _EMPTY_HELP_SCRIPT)

    # When
    result = run_xtb_path(run_dir, _config(fake), charge=0, uhf=0)

    # Then: the seed probe degrades to "not supported", never fabricates one
    assert result.returncode == 0
    assert result.seed_supported is False
    assert result.seed is None
    assert "--seed" not in result.argv
