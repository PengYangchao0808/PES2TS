"""xTB PATH subprocess runner for stage G2.

This module owns exactly one concern: driving the external GFN2-xTB binary
over a prepared run directory and recording what happened.  The command
form is frozen by the plan contract::

    <exe> start.xyz --path end.xyz --input path.inp -P <threads> --gfn 2 \
        --chrg <q> --uhf <u> [*extra_args]

with ``cwd`` set to the run directory, ``OMP_NUM_THREADS=<threads>`` in the
environment, and stdout+stderr merged into ``xtb_path.log``.  The run
directory is the only place this module writes.  Validity judgement,
output parsing, and reverse-retry policy belong to the pipeline, not here.

xTB PATH metadynamics is not byte-reproducible, so the returned
:class:`XtbRunResult` records the full provenance (argv, environment,
executable digest, version line, random-seed support) instead of promising
reproducible output.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from pes2ts_core.utils.hashing import sha256_file

#: Merged stdout+stderr log written inside the run directory.
XTB_LOG_FILENAME: Final[str] = "xtb_path.log"
#: xTB ``$path`` control block written inside the run directory.
PATH_INP_FILENAME: Final[str] = "path.inp"
#: Fixed input geometry names of the frozen command contract.
START_XYZ_FILENAME: Final[str] = "start.xyz"
END_XYZ_FILENAME: Final[str] = "end.xyz"

#: ``$path`` block parameters in write order, with the plan-default values
#: used for any key the ``g2.path`` config section does not set.
PATH_PARAM_DEFAULTS: Final[Mapping[str, Any]] = MappingProxyType(
    {
        "nrun": 1,
        "npoint": 25,
        "anopt": 10,
        "kpush": 0.003,
        "kpull": -0.015,
        "ppull": 0.05,
        "alp": 1.2,
    }
)
PATH_PARAM_ORDER: Final[tuple[str, ...]] = (
    "nrun",
    "npoint",
    "anopt",
    "kpush",
    "kpull",
    "ppull",
    "alp",
)

#: A random-seed flag as advertised by ``xtb --help`` (for example
#: ``--seed`` or ``--random-seed``); xTB 6.7.1 advertises none.
_SEED_FLAG_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?<![\w-])(--(?:[a-z][a-z0-9-]*-)?seed)\b"
)
#: Wall-clock budget for the single ``--help`` probe.
_HELP_PROBE_TIMEOUT_SECONDS: Final[float] = 30.0


class XtbNotFoundError(RuntimeError):
    """Raised when no configured candidate resolves to an xTB executable."""


def _is_executable_file(candidate: Path) -> bool:
    return candidate.is_file() and os.access(candidate, os.X_OK)


def resolve_executable(config: Mapping[str, Any]) -> Path:
    """Return the xTB binary to run, honoring the frozen precedence chain.

    An explicitly configured ``g2.xtb.executable`` must exist and be
    executable — a missing explicit setting is an error, never a silent
    fallback to a different binary.  Otherwise ``xtb`` is looked up on
    ``PATH`` and finally in ``g2.xtb.fallback_paths``.
    """
    xtb_cfg: Mapping[str, Any] = config.get("g2", {}).get("xtb", {})
    configured = xtb_cfg.get("executable")
    if configured is not None:
        candidate = Path(str(configured))
        if _is_executable_file(candidate):
            return candidate.resolve()
        raise XtbNotFoundError(
            f"g2.xtb.executable is set to {configured!r} but that path is not "
            "an executable file; fix or clear the setting in the config"
        )
    tried: list[str] = []
    found = shutil.which("xtb")
    if found is not None:
        return Path(found).resolve()
    tried.append('shutil.which("xtb") -> not found')
    for fallback in xtb_cfg.get("fallback_paths", ()):
        candidate = Path(str(fallback))
        if _is_executable_file(candidate):
            return candidate.resolve()
        tried.append(f"fallback {fallback} (missing or not executable)")
    raise XtbNotFoundError(
        "No xTB executable found. Install GFN2-xTB and put it on PATH, or "
        "set g2.xtb.executable in the config to the binary location. "
        "Tried:\n  " + "\n  ".join(tried)
    )


def write_path_inp(path: str | Path, config: Mapping[str, Any]) -> None:
    """Write the ``$path`` control block to *path* from ``g2.path``.

    Keys missing from the config fall back to :data:`PATH_PARAM_DEFAULTS`.
    The parent directory must already exist (the run directory).
    """
    path_cfg: Mapping[str, Any] = config.get("g2", {}).get("path", {})
    lines = ["$path"]
    for key in PATH_PARAM_ORDER:
        lines.append(f"   {key}={path_cfg.get(key, PATH_PARAM_DEFAULTS[key])}")
    lines.append("$end")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


@dataclass(frozen=True, slots=True)
class XtbRunResult:
    """Provenance-complete record of one xTB PATH subprocess run."""

    #: The exact argv handed to the process.
    argv: tuple[str, ...]
    #: The full environment handed to the process.
    env: Mapping[str, str]
    #: Resolved executable path and its sha256 fingerprint.
    executable: str
    executable_sha256: str
    #: First ``xtb version ...`` log line, or None when none was printed.
    xtb_version_line: str | None
    #: Exit code; negative values encode death by signal (e.g. SIGKILL).
    returncode: int | None
    #: True when the run exceeded the configured timeout and was killed.
    timed_out: bool
    #: Wall-clock duration of the managed subprocess in seconds.
    duration_seconds: float
    #: Whether ``--help`` advertised a random-seed flag that was passed.
    seed_supported: bool
    #: The seed passed (None when the binary does not support one).
    seed: int | None
    #: Path of the merged stdout+stderr log inside the run directory.
    log_path: str


def _probe_seed_flag(
    executable: Path, env: Mapping[str, str], cwd: Path
) -> str | None:
    """Return the random-seed flag advertised by ``<exe> --help``, or None.

    Any probe failure (non-zero exit, timeout, OSError, silent or
    unparseable help) degrades to "not supported".
    """
    try:
        probe = subprocess.run(
            [str(executable), "--help"],
            cwd=cwd,
            env=dict(env),
            capture_output=True,
            timeout=_HELP_PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if probe.returncode != 0:
        return None
    help_text = probe.stdout.decode("utf-8", errors="replace")
    match = _SEED_FLAG_PATTERN.search(help_text)
    return match.group(1) if match else None


def _find_version_line(log_text: str) -> str | None:
    for line in log_text.splitlines():
        if "xtb version" in line:
            return line.strip()
    return None


def run_xtb_path(
    run_dir: str | Path,
    config: Mapping[str, Any],
    *,
    charge: int = 0,
    uhf: int = 0,
    extra_args: Sequence[str] = (),
) -> XtbRunResult:
    """Run the frozen xTB PATH command inside *run_dir* and record it.

    Writes ``path.inp`` and ``xtb_path.log`` into *run_dir* (which must
    already contain ``start.xyz``/``end.xyz``), spawns the binary in its
    own session so a timeout can kill the whole process group, and never
    judges validity — the pipeline owns that decision.
    """
    directory = Path(run_dir)
    xtb_cfg: Mapping[str, Any] = config.get("g2", {}).get("xtb", {})
    threads = int(xtb_cfg.get("threads", 4))
    timeout_seconds = float(xtb_cfg.get("timeout_seconds", 1800))
    seed = int(xtb_cfg.get("seed", 42))
    executable = resolve_executable(config)
    write_path_inp(directory / PATH_INP_FILENAME, config)
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    seed_flag = _probe_seed_flag(executable, env, directory)
    argv = [
        str(executable),
        START_XYZ_FILENAME,
        "--path",
        END_XYZ_FILENAME,
        "--input",
        PATH_INP_FILENAME,
        "-P",
        str(threads),
        "--gfn",
        "2",
        "--chrg",
        str(charge),
        "--uhf",
        str(uhf),
    ]
    if seed_flag is not None:
        argv.extend([seed_flag, str(seed)])
    argv.extend(extra_args)
    log_path = directory / XTB_LOG_FILENAME
    started = time.monotonic()
    with log_path.open("wb") as log_handle:
        process = subprocess.Popen(
            argv,
            cwd=directory,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            process.wait(timeout=timeout_seconds)
            timed_out = False
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass  # the whole process group already exited
            process.wait()
        duration_seconds = time.monotonic() - started
    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    return XtbRunResult(
        argv=tuple(argv),
        env=env,
        executable=str(executable),
        executable_sha256=sha256_file(executable),
        xtb_version_line=_find_version_line(log_text),
        returncode=process.returncode,
        timed_out=timed_out,
        duration_seconds=duration_seconds,
        seed_supported=seed_flag is not None,
        seed=seed if seed_flag is not None else None,
        log_path=str(log_path),
    )


__all__ = [
    "END_XYZ_FILENAME",
    "PATH_INP_FILENAME",
    "PATH_PARAM_DEFAULTS",
    "START_XYZ_FILENAME",
    "XTB_LOG_FILENAME",
    "XtbNotFoundError",
    "XtbRunResult",
    "resolve_executable",
    "run_xtb_path",
    "write_path_inp",
]
