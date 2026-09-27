"""Tests for the xTB PATH output parsers (plan task 6).

Real-output assertions are pinned to the xTB 6.7.1 run documented in
``tests/fixtures/g2/real_path/PROVENANCE.md``. Synthetic malformed inputs are
always written to ``tmp_path``; the real fixture is never modified.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from pes2ts_core.g2.xtb_output import (
    XtbOutputError,
    check_npath_consistency,
    enumerate_trial_segments,
    parse_path_log,
    parse_path_xyz,
    parse_start_xyz,
    parse_ts_xyz,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "g2" / "real_path"

#: The 15-atom fixture system: 6 C, 1 O, 8 H in xTB output order.
EXPECTED_ELEMENTS: tuple[str, ...] = ("C", "O", "C", "C", "C", "C", "C") + ("H",) * 8

#: Minimal synthetic 3-atom geometry used to build malformed files.
ATOMS: tuple[tuple[str, float, float, float], ...] = (
    ("C", 0.0, 0.0, 0.0),
    ("H", 1.0, 0.0, 0.0),
    ("O", 0.0, 1.0, 0.0),
)
COMMENT = " energy: -1.5 xtb: 6.7.1 (test)"

_LOG_TEMPLATE = (
    " forward  barrier (kcal)  :    10.364\n"
    " backward barrier (kcal)  :    35.439\n"
    " reaction energy  (kcal)  :   -25.075\n"
    " npath : {npath}\n"
)


def _atom_lines(atoms: tuple[tuple[str, float, float, float], ...]) -> list[str]:
    return [f"{element} {x:.6f} {y:.6f} {z:.6f}" for element, x, y, z in atoms]


def _write_xyz(
    path: Path,
    frames: list[tuple[str, tuple[tuple[str, float, float, float], ...]]],
) -> None:
    lines: list[str] = []
    for comment, atoms in frames:
        lines.append(str(len(atoms)))
        lines.append(comment)
        lines.extend(_atom_lines(atoms))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _log_table_rows() -> list[tuple[int, float]]:
    """Parse the ``path data`` table (point, energy) from the real log."""
    row = re.compile(r"^\s*(\d+)\s+[-0-9.]+\s+(-?[0-9.]+)\s+")
    rows: list[tuple[int, float]] = []
    in_table = False
    for line in (FIXTURES / "xtb_path.log").read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("point"):
            in_table = True
            continue
        if in_table:
            match = row.match(line)
            if match is None:
                break
            rows.append((int(match.group(1)), float(match.group(2))))
    return rows


# ---------------------------------------------------------------------------
# Real-fixture happy paths
# ---------------------------------------------------------------------------


def test_parse_path_xyz_pinned_frame_count_and_energies() -> None:
    frames = parse_path_xyz(FIXTURES / "xtbpath.xyz")
    assert len(frames) == 22
    assert frames[0].energy == pytest.approx(-0.000000, abs=1e-6)
    assert frames[-1].energy == pytest.approx(-25.074971, abs=1e-6)


def test_parse_path_xyz_frames_consistent_with_each_other_and_start() -> None:
    frames = parse_path_xyz(FIXTURES / "xtbpath.xyz", start_path=FIXTURES / "start.xyz")
    start = parse_start_xyz(FIXTURES / "start.xyz")
    assert start.elements == EXPECTED_ELEMENTS
    assert start.energy is None
    for frame in frames:
        assert frame.elements == EXPECTED_ELEMENTS
        assert len(frame.coordinates) == 15
    assert frames[0].coordinates[0] == pytest.approx(
        (-1.05403099442019, -0.85921109056983, -1.07844127989077), abs=1e-12
    )


def test_parse_ts_xyz_energy_and_atoms() -> None:
    ts = parse_ts_xyz(FIXTURES / "xtbpath_ts.xyz")
    assert ts.energy == pytest.approx(-20.902910, abs=1e-6)
    assert ts.elements == EXPECTED_ELEMENTS
    assert len(ts.coordinates) == 15


def test_parse_path_log_summary_fields() -> None:
    log = parse_path_log(FIXTURES / "xtb_path.log")
    assert log.forward_barrier_kcal == pytest.approx(10.364, abs=1e-6)
    assert log.backward_barrier_kcal == pytest.approx(35.439, abs=1e-6)
    assert log.reaction_energy_kcal == pytest.approx(-25.075, abs=1e-6)
    # xTB 6.7.1 prints no npath token: the field stays None, never fabricated.
    assert log.npath is None


def test_frame_energies_match_the_log_path_table() -> None:
    frames = parse_path_xyz(FIXTURES / "xtbpath.xyz")
    rows = _log_table_rows()
    assert len(rows) == 21  # log table covers points 2..22
    for point, energy in rows:
        assert frames[point - 1].energy == pytest.approx(energy, abs=5e-4)


def test_log_barriers_are_consistent_with_frame_energies() -> None:
    frames = parse_path_xyz(FIXTURES / "xtbpath.xyz")
    log = parse_path_log(FIXTURES / "xtb_path.log")
    energies = [frame.energy for frame in frames]
    assert log.forward_barrier_kcal is not None
    assert log.forward_barrier_kcal == pytest.approx(max(energies), abs=5e-4)
    assert log.reaction_energy_kcal == pytest.approx(energies[-1], abs=5e-4)
    assert log.backward_barrier_kcal == pytest.approx(
        log.forward_barrier_kcal - log.reaction_energy_kcal, abs=5e-4
    )


def test_enumerate_trial_segments_registers_only_xtbpath_n_files(
    tmp_path: Path,
) -> None:
    # Trial segments are registered by filename only and never parsed:
    # garbage content must not matter.
    for name in ("xtbpath_0.xyz", "xtbpath_1.xyz", "xtbpath_10.xyz"):
        (tmp_path / name).write_text("garbage\n", encoding="utf-8")
    (tmp_path / "xtbpath_ts.xyz").write_text("3\nC 0 0 0\n", encoding="utf-8")
    (tmp_path / "xtbpath.xyz").write_text("3\nC 0 0 0\n", encoding="utf-8")
    (tmp_path / "xtbpath_2.xyz").mkdir()
    segments = enumerate_trial_segments(tmp_path)
    assert [entry.name for entry in segments] == [
        "xtbpath_0.xyz",
        "xtbpath_1.xyz",
        "xtbpath_10.xyz",
    ]


# ---------------------------------------------------------------------------
# Failure cases (synthetic files in tmp_path)
# ---------------------------------------------------------------------------


def test_missing_ts_file_raises_with_detail(tmp_path: Path) -> None:
    with pytest.raises(XtbOutputError) as excinfo:
        parse_ts_xyz(tmp_path / "xtbpath_ts.xyz")
    assert excinfo.value.detail
    assert "xtbpath_ts.xyz" in excinfo.value.detail


def test_inconsistent_frame_header_atom_count_raises(tmp_path: Path) -> None:
    path = tmp_path / "xtbpath.xyz"
    good = "\n".join([str(len(ATOMS)), COMMENT, *_atom_lines(ATOMS)])
    bad = "\n".join(["18", COMMENT, *_atom_lines(ATOMS)])
    path.write_text(good + "\n" + bad + "\n", encoding="utf-8")
    with pytest.raises(XtbOutputError):
        parse_path_xyz(path)


def test_missing_energy_comment_raises(tmp_path: Path) -> None:
    path = tmp_path / "xtbpath.xyz"
    _write_xyz(path, [(COMMENT, ATOMS), (" no energy here xtb: 6.7.1", ATOMS)])
    with pytest.raises(XtbOutputError) as excinfo:
        parse_path_xyz(path)
    assert "energy" in excinfo.value.detail


def test_ts_missing_energy_comment_raises(tmp_path: Path) -> None:
    path = tmp_path / "xtbpath_ts.xyz"
    _write_xyz(path, [("", ATOMS)])
    with pytest.raises(XtbOutputError) as excinfo:
        parse_ts_xyz(path)
    assert "energy" in excinfo.value.detail


def test_truncated_file_raises(tmp_path: Path) -> None:
    full = (FIXTURES / "xtbpath.xyz").read_text(encoding="utf-8")
    truncated = tmp_path / "xtbpath.xyz"
    truncated.write_text(full[: len(full) // 2], encoding="utf-8")
    with pytest.raises(XtbOutputError):
        parse_path_xyz(truncated)


def test_single_frame_path_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "xtbpath.xyz"
    _write_xyz(path, [(COMMENT, ATOMS)])
    with pytest.raises(XtbOutputError) as excinfo:
        parse_path_xyz(path)
    assert "frame" in excinfo.value.detail


def test_element_sequence_mismatch_across_frames_raises(tmp_path: Path) -> None:
    path = tmp_path / "xtbpath.xyz"
    swapped = (("N", 0.0, 0.0, 0.0), ("H", 1.0, 0.0, 0.0), ("O", 0.0, 1.0, 0.0))
    _write_xyz(path, [(COMMENT, ATOMS), (COMMENT, swapped)])
    with pytest.raises(XtbOutputError) as excinfo:
        parse_path_xyz(path)
    assert "element" in excinfo.value.detail


def test_element_sequence_inconsistent_with_start_raises(tmp_path: Path) -> None:
    start = tmp_path / "start.xyz"
    different = (("N", 0.0, 0.0, 0.0), ("H", 1.0, 0.0, 0.0), ("O", 0.0, 1.0, 0.0))
    _write_xyz(start, [("", different)])
    path = tmp_path / "xtbpath.xyz"
    _write_xyz(path, [(COMMENT, ATOMS), (COMMENT, ATOMS)])
    with pytest.raises(XtbOutputError) as excinfo:
        parse_path_xyz(path, start_path=start)
    assert "element" in excinfo.value.detail


def test_multi_frame_ts_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "xtbpath_ts.xyz"
    _write_xyz(path, [(COMMENT, ATOMS), (COMMENT, ATOMS)])
    with pytest.raises(XtbOutputError):
        parse_ts_xyz(path)


def test_non_integer_frame_header_raises(tmp_path: Path) -> None:
    path = tmp_path / "xtbpath.xyz"
    lines = ["fifteen", COMMENT, *_atom_lines(ATOMS)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(XtbOutputError):
        parse_path_xyz(path)


def test_empty_path_file_raises(tmp_path: Path) -> None:
    path = tmp_path / "xtbpath.xyz"
    path.write_text("", encoding="utf-8")
    with pytest.raises(XtbOutputError):
        parse_path_xyz(path)


def test_non_numeric_coordinate_raises(tmp_path: Path) -> None:
    path = tmp_path / "xtbpath.xyz"
    lines = ["3", COMMENT, "C 1.0 abc 0.0", "H 1.0 0.0 0.0", "O 0.0 1.0 0.0"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(XtbOutputError):
        parse_path_xyz(path)


# ---------------------------------------------------------------------------
# npath cross-check
# ---------------------------------------------------------------------------


def test_npath_mismatch_with_frame_count_raises(tmp_path: Path) -> None:
    log_path = tmp_path / "xtb_path.log"
    log_path.write_text(_LOG_TEMPLATE.format(npath=25), encoding="utf-8")
    log = parse_path_log(log_path)
    assert log.npath == 25
    frames = parse_path_xyz(FIXTURES / "xtbpath.xyz")
    with pytest.raises(XtbOutputError) as excinfo:
        check_npath_consistency(log, len(frames))
    assert "25" in excinfo.value.detail
    assert "22" in excinfo.value.detail


def test_npath_matching_frame_count_passes(tmp_path: Path) -> None:
    log_path = tmp_path / "xtb_path.log"
    log_path.write_text(_LOG_TEMPLATE.format(npath=22), encoding="utf-8")
    frames = parse_path_xyz(FIXTURES / "xtbpath.xyz")
    check_npath_consistency(parse_path_log(log_path), len(frames))


def test_npath_absent_skips_cross_check() -> None:
    log = parse_path_log(FIXTURES / "xtb_path.log")
    check_npath_consistency(log, 5)  # no npath in the log: no-op
