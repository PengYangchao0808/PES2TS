"""Equivalence-certificate tests (ADR-0002 X3'-A): RecipeEquivalence + ReplayParity.

Pinned fixture: ``tests/fixtures/g2/real_path/`` — a real 22-frame GFN2-xTB
PATH trajectory (xTB 6.7.1, 15-atom C7OH8, see ``PROVENANCE.md``) plus its
``path.inp``/``start.xyz``/``end.xyz`` endpoints.  The fixture carries NO
historical derived artifacts (no ``frames.parquet`` / ``reaction_path.json``),
so ReplayParity expectations are computed in-test from the same fixture bytes
through the pipeline derivation (``parse_path_xyz`` -> ``frame_metrics`` +
``evaluate_validity`` -> ``stable_json_dumps``) — never fabricated constants.
The recipe expectation is the real fixture ``path.inp`` text and its
PROVENANCE-pinned sha256.

Endpoint context for the replay derivation is derived the way the pipeline
does: map-ordered ``start.xyz`` (reactant) and ``end.xyz`` (product);
``r_pairs``/``p_pairs``/``events`` are perceived geometrically from those
same fixture endpoints (Cordero radii + 0.45 A), so every replay input is a
function of real fixture bytes.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pes2ts_core.g0.rp_checks import ELEMENT_SYMBOLS
from pes2ts_core.g1.index_map import COVALENT_RADII
from pes2ts_core.generation.execution.xtb_path.equivalence import (
    EQUIVALENCE_REPORT_FILENAME,
    SCHEMA_EQUIVALENCE,
    build_recipe_equivalence_certificate,
    build_replay_parity_certificate,
    write_equivalence_report,
)
from pes2ts_core.generation.execution.xtb_path.inspect import (
    evaluate_validity,
    frame_metrics,
)
from pes2ts_core.generation.execution.xtb_path.xtb_output import (
    parse_path_xyz,
    parse_start_xyz,
)
from pes2ts_core.utils.hashing import sha256_bytes, sha256_file, stable_json_dumps

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "g2" / "real_path"

#: The fixture-era ``$path`` block (PROVENANCE.md: npoint=25, alp=1.2).
FIXTURE_PATH_CONFIG = {
    "nrun": 1,
    "npoint": 25,
    "anopt": 10,
    "kpush": 0.003,
    "kpull": -0.015,
    "ppull": 0.05,
    "alp": 1.2,
}
#: PROVENANCE-pinned sha256 of the fixture ``path.inp`` bytes.
FIXTURE_PATH_INP_SHA256 = (
    "5932fe20a55e06e9d79883f6b72d989fedab2921403770caf904a94bf8b082a1"
)
REPLAY_REACTION_ID = "RXN_REPLAY_FIXTURE"
#: Bond tolerance mirroring ``g2.assembly.bond_tolerance`` default.
BOND_TOLERANCE = 0.45


def _radius(element: str) -> float:
    return COVALENT_RADII[ELEMENT_SYMBOLS.index(element) + 1]


def _keyed(frame) -> dict[int, tuple[float, float, float]]:
    return {index + 1: xyz for index, xyz in enumerate(frame.coordinates)}


def _bonds(coords: dict[int, tuple[float, float, float]], elements: dict[int, str]):
    pairs = set()
    maps = sorted(coords)
    for position, first in enumerate(maps):
        for second in maps[position + 1 :]:
            distance = math.dist(coords[first], coords[second])
            if distance < _radius(elements[first]) + _radius(elements[second]) + BOND_TOLERANCE:
                pairs.add((first, second))
    return frozenset(pairs)


def _partner(hydrogen: int, pairs) -> int | None:
    for first, second in pairs:
        if first == hydrogen:
            return second
        if second == hydrogen:
            return first
    return None


def _fixture_replay_context():
    """Replay inputs derived from the real fixture endpoints (never invented)."""
    start = parse_start_xyz(FIXTURES / "start.xyz")
    end = parse_start_xyz(FIXTURES / "end.xyz")
    elements = {index + 1: el for index, el in enumerate(start.elements)}
    reactant = _keyed(start)
    product = _keyed(end)
    r_pairs = _bonds(reactant, elements)
    p_pairs = _bonds(product, elements)
    h_migration = [
        {"h": map_, "from": _partner(map_, r_pairs), "to": _partner(map_, p_pairs)}
        for map_, element in sorted(elements.items())
        if element == "H" and _partner(map_, r_pairs) != _partner(map_, p_pairs)
    ]
    events = {
        "formed": [
            {"atoms": list(pair), "order_r": None, "order_p": 1.0}
            for pair in sorted(p_pairs - r_pairs)
        ],
        "broken": [
            {"atoms": list(pair), "order_r": 1.0, "order_p": None}
            for pair in sorted(r_pairs - p_pairs)
        ],
        "order_changed": [],
        "hydrogen_migration": h_migration,
    }
    return r_pairs, p_pairs, events


def _direct_replay():
    """Independent pipeline derivation over the fixture (test-side oracle)."""
    r_pairs, p_pairs, events = _fixture_replay_context()
    frames = parse_path_xyz(FIXTURES / "xtbpath.xyz", start_path=FIXTURES / "start.xyz")
    start_frame = parse_start_xyz(FIXTURES / "start.xyz")
    product_frame = parse_start_xyz(FIXTURES / "end.xyz")
    maps = tuple(range(1, len(start_frame.coordinates) + 1))
    elements = {map_: start_frame.elements[i] for i, map_ in enumerate(maps)}
    reactant = {map_: start_frame.coordinates[i] for i, map_ in enumerate(maps)}
    product = {map_: product_frame.coordinates[i] for i, map_ in enumerate(maps)}
    rows = frame_metrics(
        frames,
        reaction_id=REPLAY_REACTION_ID,
        reactant_coords=reactant,
        product_coords=product,
        elements=elements,
        r_pairs=r_pairs,
        p_pairs=p_pairs,
        events=events,
        config={},
    )
    for row in rows:
        energy = row["energy_rel_kcal"]
        row["energy_rel_kcal_raw"] = None if energy is None else float(energy)
    verdict = evaluate_validity(
        frames,
        reaction_id=REPLAY_REACTION_ID,
        reactant_coords=reactant,
        product_coords=product,
        elements=elements,
        r_pairs=r_pairs,
        p_pairs=p_pairs,
        events=events,
        xtb_failure=None,
        config={},
    )
    return frames, rows, verdict


def _expected_digests(rows, verdict) -> tuple[str, str]:
    """Digests of the deterministic row/verdict serialization (test oracle)."""
    frames_sha = sha256_bytes(stable_json_dumps(list(rows)).encode("utf-8"))
    document = {
        "reaction_id": REPLAY_REACTION_ID,
        "status": str(verdict.status),
        "failure_code": (
            None if verdict.failure_code is None else verdict.failure_code.value
        ),
        "failure_detail": verdict.detail,
        "n_frames": len(rows),
        "validity": dict(verdict.summary),
    }
    document_sha = sha256_bytes(stable_json_dumps(document).encode("utf-8"))
    return frames_sha, document_sha


def _replay_kwargs(**overrides):
    r_pairs, p_pairs, events = _fixture_replay_context()
    kwargs = {
        "frozen_trajectory_path": FIXTURES / "xtbpath.xyz",
        "start_xyz_path": FIXTURES / "start.xyz",
        "charge": 0,
        "uhf": 0,
        "r_pairs": r_pairs,
        "p_pairs": p_pairs,
        "events": events,
        "config": {},
        "reaction_id": REPLAY_REACTION_ID,
        "product_xyz_path": FIXTURES / "end.xyz",
    }
    kwargs.update(overrides)
    return kwargs


def _mutated_trajectory_text(text: str) -> str:
    """Perturb one coordinate of a middle frame; the file stays parseable."""
    lines = text.splitlines()
    headers = [index for index, line in enumerate(lines) if line.strip().isdigit()]
    assert len(headers) >= 3, "fixture trajectory must hold several frames"
    target = headers[2] + 2  # first atom line of the third frame
    parts = lines[target].split()
    parts[1] = f"{float(parts[1]) + 0.75:.8f}"
    lines[target] = f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}"
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# RecipeEquivalence
# ---------------------------------------------------------------------------


def test_recipe_certificate_passes_for_frozen_fixture_config():
    expected_text = (FIXTURES / "path.inp").read_text(encoding="utf-8")
    certificate = build_recipe_equivalence_certificate(
        FIXTURE_PATH_CONFIG, expected_path_inp=expected_text
    )
    assert certificate["passed"] is True
    assert certificate["mismatch"] is None
    assert certificate["path_inp_sha256"] == FIXTURE_PATH_INP_SHA256
    assert certificate["path_inp_sha256"] == sha256_file(FIXTURES / "path.inp")
    assert certificate["expected_path_inp_sha256"] == FIXTURE_PATH_INP_SHA256
    assert certificate["config"] == FIXTURE_PATH_CONFIG


def test_recipe_certificate_fails_on_mutated_path_inp():
    expected_text = (FIXTURES / "path.inp").read_text(encoding="utf-8")
    mutated = expected_text.replace("npoint=25", "npoint=50")
    assert mutated != expected_text
    certificate = build_recipe_equivalence_certificate(
        FIXTURE_PATH_CONFIG, expected_path_inp=mutated
    )
    assert certificate["passed"] is False
    assert certificate["mismatch"] is not None
    assert "path_inp byte mismatch" in certificate["mismatch"]
    assert certificate["path_inp_sha256"] != certificate["expected_path_inp_sha256"]


def test_recipe_certificate_fails_when_config_diverges_from_frozen_text():
    # Default config renders npoint=50/alp=0.5; the fixture text is 25/1.2.
    expected_text = (FIXTURES / "path.inp").read_text(encoding="utf-8")
    certificate = build_recipe_equivalence_certificate({}, expected_path_inp=expected_text)
    assert certificate["passed"] is False
    assert certificate["mismatch"] is not None


def test_recipe_certificate_request_sha256_anchor_passes_and_fails():
    expected_text = (FIXTURES / "path.inp").read_text(encoding="utf-8")
    good = build_recipe_equivalence_certificate(
        FIXTURE_PATH_CONFIG,
        expected_path_inp=expected_text,
        expected_request_sha256=FIXTURE_PATH_INP_SHA256,
    )
    assert good["passed"] is True
    bad = build_recipe_equivalence_certificate(
        FIXTURE_PATH_CONFIG,
        expected_path_inp=expected_text,
        expected_request_sha256="0" * 64,
    )
    assert bad["passed"] is False
    assert bad["mismatch"] is not None
    assert "request_sha256 mismatch" in bad["mismatch"]


def test_recipe_certificate_never_raises_on_non_mapping_config():
    certificate = build_recipe_equivalence_certificate([("nrun", 1)])  # type: ignore[arg-type]
    assert certificate["passed"] is False
    assert certificate["path_inp_sha256"] is None
    assert certificate["mismatch"] is not None


# ---------------------------------------------------------------------------
# ReplayParity
# ---------------------------------------------------------------------------


def test_replay_fixture_has_no_historical_derived_artifacts():
    # The pinned fixture ships raw xTB outputs only; ReplayParity expectations
    # are therefore computed in-test from the same fixture bytes (never from
    # fabricated constants, never from missing historical artifacts).
    assert not (FIXTURES / "frames.parquet").exists()
    assert not (FIXTURES / "reaction_path.json").exists()
    assert (FIXTURES / "xtbpath.xyz").is_file()
    assert (FIXTURES / "start.xyz").is_file()
    assert (FIXTURES / "end.xyz").is_file()


def test_replay_certificate_passes_against_test_derived_digests():
    frames, rows, verdict = _direct_replay()
    frames_sha, document_sha = _expected_digests(rows, verdict)
    certificate = build_replay_parity_certificate(
        **_replay_kwargs(
            expected_frames_sha256=frames_sha,
            expected_document_sha256=document_sha,
        )
    )
    assert certificate["passed"] is True
    assert certificate["mismatch"] is None
    assert certificate["n_frames"] == len(frames) == 22
    assert certificate["frames_sha256"] == frames_sha
    assert certificate["document_sha256"] == document_sha


def test_replay_certificate_is_deterministic_across_calls():
    first = build_replay_parity_certificate(**_replay_kwargs())
    second = build_replay_parity_certificate(**_replay_kwargs())
    assert first["frames_sha256"] == second["frames_sha256"]
    assert first["document_sha256"] == second["document_sha256"]
    assert first["n_frames"] == second["n_frames"] == 22
    consistent = build_replay_parity_certificate(
        **_replay_kwargs(
            expected_frames_sha256=first["frames_sha256"],
            expected_document_sha256=first["document_sha256"],
        )
    )
    assert consistent["passed"] is True


def test_replay_certificate_fails_on_mutated_trajectory(tmp_path: Path):
    _, rows, verdict = _direct_replay()
    frames_sha, document_sha = _expected_digests(rows, verdict)
    mutated = tmp_path / "xtbpath.xyz"
    mutated.write_text(
        _mutated_trajectory_text(
            (FIXTURES / "xtbpath.xyz").read_text(encoding="utf-8")
        ),
        encoding="utf-8",
    )
    certificate = build_replay_parity_certificate(
        **_replay_kwargs(
            frozen_trajectory_path=mutated,
            expected_frames_sha256=frames_sha,
            expected_document_sha256=document_sha,
        )
    )
    assert certificate["passed"] is False
    assert certificate["mismatch"] is not None
    assert "frames digest mismatch" in certificate["mismatch"]
    assert certificate["frames_sha256"] != frames_sha
    assert certificate["n_frames"] == 22


def test_replay_certificate_reports_unparseable_trajectory_without_raising(
    tmp_path: Path,
):
    broken = tmp_path / "xtbpath.xyz"
    broken.write_text("not a trajectory\n", encoding="utf-8")
    certificate = build_replay_parity_certificate(
        **_replay_kwargs(frozen_trajectory_path=broken)
    )
    assert certificate["passed"] is False
    assert certificate["n_frames"] == 0
    assert certificate["frames_sha256"] is None
    assert certificate["mismatch"] is not None
    assert "replay derivation failed" in certificate["mismatch"]


def test_replay_certificate_derives_product_endpoint_from_sibling_end_xyz():
    # No product_xyz_path/product_coords: the fixture keeps end.xyz beside
    # xtbpath.xyz, and the sibling fallback reproduces the explicit call.
    _, rows, verdict = _direct_replay()
    frames_sha, document_sha = _expected_digests(rows, verdict)
    certificate = build_replay_parity_certificate(
        **_replay_kwargs(
            product_xyz_path=None,
            expected_frames_sha256=frames_sha,
            expected_document_sha256=document_sha,
        )
    )
    assert certificate["passed"] is True
    assert certificate["frames_sha256"] == frames_sha


# ---------------------------------------------------------------------------
# equivalence_report.json
# ---------------------------------------------------------------------------


def test_write_equivalence_report_produces_expected_json_keys(tmp_path: Path):
    expected_text = (FIXTURES / "path.inp").read_text(encoding="utf-8")
    recipe = build_recipe_equivalence_certificate(
        FIXTURE_PATH_CONFIG, expected_path_inp=expected_text
    )
    _, rows, verdict = _direct_replay()
    frames_sha, document_sha = _expected_digests(rows, verdict)
    replay = build_replay_parity_certificate(
        **_replay_kwargs(
            expected_frames_sha256=frames_sha,
            expected_document_sha256=document_sha,
        )
    )
    report_path = tmp_path / EQUIVALENCE_REPORT_FILENAME
    write_equivalence_report(report_path, recipe=recipe, replay=replay)
    document = json.loads(report_path.read_text(encoding="utf-8"))
    assert set(document) == {"schema_version", "recipe_equivalence", "replay_parity"}
    assert document["schema_version"] == SCHEMA_EQUIVALENCE
    assert document["recipe_equivalence"] == recipe
    assert document["replay_parity"] == replay
    assert set(document["recipe_equivalence"]) == {
        "passed",
        "path_inp_sha256",
        "expected_path_inp_sha256",
        "mismatch",
        "config",
    }
    assert set(document["replay_parity"]) == {
        "passed",
        "n_frames",
        "frames_sha256",
        "expected_frames_sha256",
        "document_sha256",
        "expected_document_sha256",
        "mismatch",
    }
    assert document["recipe_equivalence"]["passed"] is True
    assert document["replay_parity"]["passed"] is True


def test_write_equivalence_report_is_byte_stable_and_accepts_a_directory(
    tmp_path: Path,
):
    recipe = build_recipe_equivalence_certificate(FIXTURE_PATH_CONFIG)
    replay = build_replay_parity_certificate(**_replay_kwargs())
    report_dir = tmp_path / "reports"
    report_dir.mkdir()
    write_equivalence_report(report_dir, recipe=recipe, replay=replay)
    report_path = report_dir / EQUIVALENCE_REPORT_FILENAME
    first_bytes = report_path.read_bytes()
    write_equivalence_report(report_path, recipe=recipe, replay=replay)
    assert report_path.read_bytes() == first_bytes
    document = json.loads(first_bytes.decode("utf-8"))
    assert document["recipe_equivalence"]["passed"] is True
    assert document["replay_parity"]["passed"] is True
