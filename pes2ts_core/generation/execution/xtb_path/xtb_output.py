"""Parsers for real xTB PATH outputs: path frames, TS guess, and the log.

Contract (plan task 6):

- ``xtbpath.xyz`` is the ONLY authoritative frame source. ``xtbpath_<n>.xyz``
  files are trial segments: never parsed for frames, only registered by
  :func:`enumerate_trial_segments` (they are ACP-owned RESULT artifacts; no
  PES2TS config switch retains or deletes them).
- Comment energies (`` energy: <value> ...``) are RELATIVE kcal/mol (first
  frame ~0, later frames may be negative), never absolute total energies. A
  comment energy must be a decimal number where required (path frames, TS
  guess); ``NaN``/``inf`` comments are unparseable. Coordinates are finite.
- :func:`parse_path_log` extracts forward/backward barrier, reaction energy,
  and ``npath``; a missing field is ``None``, never fabricated (xTB 6.7.1
  prints no ``npath`` token, so the real-log value is ``None``).
- No validity thresholds are judged here (that is ``g2.inspect``); this layer
  only rejects structurally malformed, truncated, or inconsistent output via
  :class:`XtbOutputError` whose ``detail`` names file, frame index, problem.
- Pure functions: paths in, data out. xTB is never invoked here.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

#: Trial segment filenames written by xTB 6.7.1 (never authoritative).
_TRIAL_SEGMENT_PATTERN = re.compile(r"^xtbpath_(\d+)\.xyz$")

#: A decimal float literal as written by xTB in comments and log summaries.
_NUMBER = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?)"

#: `` energy: 0.000000000000 xtb: 6.7.1 (edcfbbe)`` comment lines.
_ENERGY_PATTERN = re.compile(r"energy:\s*" + _NUMBER)

_FORWARD_BARRIER_PATTERN = re.compile(
    r"forward\s+barrier\s*\(kcal\)\s*:\s*" + _NUMBER, re.IGNORECASE
)
_BACKWARD_BARRIER_PATTERN = re.compile(
    r"backward\s+barrier\s*\(kcal\)\s*:\s*" + _NUMBER, re.IGNORECASE
)
_REACTION_ENERGY_PATTERN = re.compile(
    r"reaction\s+energy\s*\(kcal\)\s*:\s*" + _NUMBER, re.IGNORECASE
)
#: An explicit ``npath`` count line (absent in xTB 6.7.1 logs).
_NPATH_PATTERN = re.compile(r"^\s*npath\b\s*[:=]?\s*(\d+)\s*$", re.MULTILINE)


class XtbOutputError(ValueError):
    """A real xTB output file is missing, malformed, or inconsistent.

    ``detail`` is a machine-readable string naming the file, the frame index
    when applicable, and the concrete problem; the message is the detail.
    """

    def __init__(self, detail: str) -> None:
        super().__init__(detail)
        self.detail = detail


@dataclass(frozen=True, slots=True)
class Frame:
    """One parsed XYZ frame.

    ``elements`` and ``coordinates`` are index-aligned; ``energy`` is the
    relative kcal/mol comment value, or ``None`` when the comment carries no
    energy (only allowed for plain endpoint files such as ``start.xyz``).
    """

    elements: tuple[str, ...]
    coordinates: tuple[tuple[float, float, float], ...]
    energy: float | None


@dataclass(frozen=True, slots=True)
class PathLog:
    """Human-readable summary of ``xtb_path.log``; ``None`` = absent in log."""

    forward_barrier_kcal: float | None
    backward_barrier_kcal: float | None
    reaction_energy_kcal: float | None
    npath: int | None


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise XtbOutputError(f"cannot read {path}: {exc}") from exc


def _parse_energy(path: Path, frame_index: int, comment: str, *, required: bool) -> float | None:
    match = _ENERGY_PATTERN.search(comment)
    if match is None:
        if required:
            raise XtbOutputError(
                f"{path}: frame {frame_index}: comment line has no parseable 'energy:' value: {comment!r}"
            )
        return None
    return float(match.group(1))


def _parse_atom(
    path: Path, frame_index: int, atom_index: int, line: str
) -> tuple[str, tuple[float, float, float]]:
    parts = line.split()
    if len(parts) < 4:
        raise XtbOutputError(
            f"{path}: frame {frame_index} atom {atom_index}: expected 'element x y z', got {line!r}"
        )
    values: list[float] = []
    for token in parts[1:4]:
        try:
            value = float(token)
        except ValueError:
            raise XtbOutputError(
                f"{path}: frame {frame_index} atom {atom_index}: coordinate {token!r} is not a number"
            ) from None
        if not math.isfinite(value):
            raise XtbOutputError(
                f"{path}: frame {frame_index} atom {atom_index}: coordinate {token!r} is not finite"
            )
        values.append(value)
    return parts[0], (values[0], values[1], values[2])


def _parse_frames(path: Path, *, require_energy: bool) -> tuple[Frame, ...]:
    """Strict multi-frame XYZ parse: whitespace tolerance only, no fallbacks."""
    lines = _read_text(path).splitlines()
    frames: list[Frame] = []
    pos = 0
    while pos < len(lines):
        if not lines[pos].strip():
            pos += 1
            continue
        header = lines[pos].strip()
        try:
            n_atoms = int(header)
        except ValueError:
            raise XtbOutputError(
                f"{path}: frame {len(frames)}: atom-count header {header!r} is not an integer"
            ) from None
        if n_atoms <= 0:
            raise XtbOutputError(
                f"{path}: frame {len(frames)}: atom-count header {header!r} is not positive"
            )
        pos += 1
        if pos >= len(lines):
            raise XtbOutputError(f"{path}: frame {len(frames)}: comment line missing (truncated)")
        energy = _parse_energy(path, len(frames), lines[pos], required=require_energy)
        pos += 1
        if pos + n_atoms > len(lines):
            raise XtbOutputError(
                f"{path}: frame {len(frames)}: header claims {n_atoms} atoms but the file ends early"
            )
        elements: list[str] = []
        coordinates: list[tuple[float, float, float]] = []
        for atom_index in range(n_atoms):
            element, xyz = _parse_atom(path, len(frames), atom_index, lines[pos + atom_index])
            elements.append(element)
            coordinates.append(xyz)
        pos += n_atoms
        frames.append(Frame(tuple(elements), tuple(coordinates), energy))
    if not frames:
        raise XtbOutputError(f"{path}: no frames found (empty file)")
    return tuple(frames)


def _check_self_consistent(path: Path, frames: tuple[Frame, ...]) -> None:
    first = frames[0]
    for index, frame in enumerate(frames[1:], start=1):
        if len(frame.elements) != len(first.elements):
            raise XtbOutputError(
                f"{path}: frame {index} has {len(frame.elements)} atoms, frame 0 has {len(first.elements)}"
            )
        if frame.elements != first.elements:
            for position, (reference, actual) in enumerate(zip(first.elements, frame.elements)):
                if reference != actual:
                    raise XtbOutputError(
                        f"{path}: frame {index} element at position {position} is {actual!r}, "
                        f"frame 0 has {reference!r}"
                    )


def parse_path_xyz(path: Path, start_path: Path | None = None) -> tuple[Frame, ...]:
    """Parse the authoritative multi-frame ``xtbpath.xyz``.

    Raises :class:`XtbOutputError` when the file is missing or malformed, has
    fewer than 2 frames, any frame's atom count or element sequence differs
    from frame 0, or — when ``start_path`` is given — differs from the
    ``start.xyz`` element sequence.
    """
    frames = _parse_frames(path, require_energy=True)
    if len(frames) < 2:
        raise XtbOutputError(f"{path}: path file has {len(frames)} frame(s); at least 2 required")
    _check_self_consistent(path, frames)
    if start_path is not None:
        start = parse_start_xyz(start_path)
        if start.elements != frames[0].elements:
            raise XtbOutputError(
                f"{path}: frame element sequence inconsistent with {start_path}: "
                f"frame 0 {frames[0].elements} vs start {start.elements}"
            )
    return frames


def parse_ts_xyz(path: Path) -> Frame:
    """Parse the single-frame TS guess ``xtbpath_ts.xyz`` (energy required)."""
    frames = _parse_frames(path, require_energy=True)
    if len(frames) != 1:
        raise XtbOutputError(f"{path}: TS guess file must contain exactly 1 frame, found {len(frames)}")
    return frames[0]


def parse_start_xyz(path: Path) -> Frame:
    """Parse the single-frame endpoint file ``start.xyz`` (no energy needed)."""
    frames = _parse_frames(path, require_energy=False)
    if len(frames) != 1:
        raise XtbOutputError(f"{path}: start file must contain exactly 1 frame, found {len(frames)}")
    return frames[0]


def _last_float(text: str, pattern: re.Pattern[str]) -> float | None:
    values = pattern.findall(text)
    return float(values[-1]) if values else None


def parse_path_log(path: Path) -> PathLog:
    """Parse the human-readable ``xtb_path.log`` summary fields.

    Missing fields are ``None`` (never fabricated; xTB 6.7.1 prints no
    ``npath`` token). When a summary appears several times, the last
    occurrence (the final run summary) wins.
    """
    text = _read_text(path)
    npath_values = _NPATH_PATTERN.findall(text)
    return PathLog(
        forward_barrier_kcal=_last_float(text, _FORWARD_BARRIER_PATTERN),
        backward_barrier_kcal=_last_float(text, _BACKWARD_BARRIER_PATTERN),
        reaction_energy_kcal=_last_float(text, _REACTION_ENERGY_PATTERN),
        npath=int(npath_values[-1]) if npath_values else None,
    )


def enumerate_trial_segments(run_dir: Path) -> tuple[Path, ...]:
    """Register ``xtbpath_<n>.xyz`` trial segment files, sorted by number.

    Trial segments are never an authoritative frame source (``xtbpath.xyz``
    is); they are ACP-owned RESULT artifacts, registered here by filename.
    The returned files must not be parsed for frames.
    """
    segments: list[tuple[int, Path]] = []
    for entry in run_dir.iterdir():
        match = _TRIAL_SEGMENT_PATTERN.match(entry.name)
        if match is not None and entry.is_file():
            segments.append((int(match.group(1)), entry))
    return tuple(entry for _, entry in sorted(segments))


def check_npath_consistency(log: PathLog, frame_count: int) -> None:
    """Cross-check a parsed ``npath`` against the authoritative frame count.

    Documented choice: a mismatch is a hard :class:`XtbOutputError` (the plan
    requires a cross-check). A log without an ``npath`` token — the xTB 6.7.1
    case — is a no-op.
    """
    if log.npath is None or log.npath == frame_count:
        return
    raise XtbOutputError(
        f"log npath={log.npath} is inconsistent with the authoritative frame count {frame_count}"
    )
