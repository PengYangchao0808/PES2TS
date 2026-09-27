"""Tests for the independent G2 tree verification (:mod:`pes2ts_core.g2.verify`)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pes2ts_core.g1.build import reaction_change_path
from pes2ts_core.g2.artifacts import (
    build_summary,
    reaction_document,
    write_frames_parquet,
    write_manifest,
    write_summary,
)
from pes2ts_core.g2.verify import G2Verification, verify_g2
from pes2ts_core.utils.hashing import sha256_bytes, sha256_file, stable_json_dumps
from pes2ts_core.utils.jsonio import write_json

REACTION_ID = "RXN_0000000001"
STRAY_ID = "RXN_0000000042"
G2_SHARD_SIZE = 1000


def _config(tmp_path: Path) -> dict[str, Any]:
    return {
        "paths": {"interim": str(tmp_path / "interim"), "manifests": str(tmp_path / "manifests")},
        "g2": {"shard_size": G2_SHARD_SIZE},
    }


def _frame_row(reaction_id: str, frame_index: int) -> dict[str, Any]:
    return {
        "reaction_id": reaction_id,
        "frame_index": frame_index,
        "energy_rel_kcal": -1.0 * frame_index,
        "energy_rel_kcal_raw": -1.0 * frame_index,
        "rmsd_to_start": 0.1 * frame_index,
        "rmsd_to_end": 0.2 * frame_index,
        "step_max": 0.05 * frame_index,
        "step_rmsd": 0.02 * frame_index,
        "min_nonbonded_distance": 1.5,
        "event_distances": {"broken:1-5": 1.2},
    }


def _write_xyz(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_clean_tree(tmp_path: Path, *, failed: bool = False) -> tuple[dict[str, Any], Path]:
    """One reaction with document, frames, endpoints, XYZ, summary, manifest.

    With ``failed=True`` the reaction is a terminal ``G2_XTB_FAILED`` failure
    that produced no parseable frames, so no ``frames.parquet`` is written.
    """
    config = _config(tmp_path)
    interim = Path(config["paths"]["interim"])
    manifests = Path(config["paths"]["manifests"])
    g1_path = reaction_change_path(interim, REACTION_ID, G2_SHARD_SIZE)
    g1_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(g1_path, {"schema_version": "g1_change_v1", "reaction_id": REACTION_ID})

    rxn_dir = interim / "g2" / "paths" / "00000" / REACTION_ID
    _write_xyz(rxn_dir / "R.xyz", "2\nRXN_0000000001 R\nC 0.0 0.0 0.0\nO 1.4 0.0 0.0\n")
    _write_xyz(rxn_dir / "P.xyz", "2\nRXN_0000000001 P\nC 0.0 0.0 0.0\nO 2.4 0.0 0.0\n")
    run_dir = rxn_dir / "run"
    sources = {
        "g1_document": sha256_file(g1_path),
        "R.xyz": sha256_file(rxn_dir / "R.xyz"),
        "P.xyz": sha256_file(rxn_dir / "P.xyz"),
        "path.inp": None,
    }
    if failed:
        _write_xyz(run_dir / "path.inp", "$path\n  maxopt=3\n$end\n")
        sources["path.inp"] = sha256_file(run_dir / "path.inp")
    else:
        _write_xyz(run_dir / "xtbpath.xyz", "2\n energy: 0.000000\nC 0.0 0.0 0.0\nO 1.4 0.0 0.0\n")
        (run_dir / "path.inp").write_text("$path\n  maxopt=3\n$end\n", encoding="utf-8")
        sources["path.inp"] = sha256_file(run_dir / "path.inp")
        sources["xtbpath.xyz"] = sha256_file(run_dir / "xtbpath.xyz")
    endpoints = {
        "reaction_id": REACTION_ID,
        "multiplicity_basis": "g1_valid_invariant",
        "reactant_atoms": [
            {"map": 1, "element": "C", "component": "R:P1", "local_index": 0, "global_index": 0,
             "coordinate": [0.0, 0.0, 0.0]},
            {"map": 5, "element": "O", "component": "R:P1", "local_index": 1, "global_index": 1,
             "coordinate": [1.4, 0.0, 0.0]},
        ],
        "product_atoms": [],
        "placements": [],
    }
    write_json(rxn_dir / "endpoints.json", endpoints)

    rows = [] if failed else [_frame_row(REACTION_ID, 0), _frame_row(REACTION_ID, 1)]
    if rows:
        write_frames_parquet(rxn_dir / "frames.parquet", rows)
    document = reaction_document(
        reaction_id=REACTION_ID,
        status="failed" if failed else "valid",
        failure_code="G2_XTB_FAILED" if failed else None,
        failure_detail="xtbpath.xyz missing" if failed else None,
        direction="forward",
        sources=sources,
        endpoints={"multiplicity_basis": "g1_valid_invariant", "n_candidates": 0},
        frames=rows,
        config_digest="0" * 64,
    )
    write_json(rxn_dir / "reaction_path.json", document)
    summary_rows = build_summary([document])
    write_summary(interim / "g2_summary.parquet", summary_rows)
    write_manifest(
        manifests / "g2_path_manifest.json",
        summary_rows,
        run_counts={"n_selected": 1, "n_attempted": 1, "n_skipped": 0},
        xtb_fingerprint={"sha256": "a" * 64, "path_inp": "$path\n"},
        config=config,
        generated_at="2026-01-01T00:00:00+00:00",
    )
    return config, rxn_dir


def _rewrite_document(rxn_dir: Path, mutate: Any) -> None:
    """Re-write reaction_path.json after applying *mutate* to the loaded document."""
    from pes2ts_core.utils.jsonio import read_json

    document = read_json(rxn_dir / "reaction_path.json")
    mutate(document)
    write_json(rxn_dir / "reaction_path.json", document)


def _rewrite_manifest(config: dict[str, Any], mutate: Any) -> None:
    from pes2ts_core.utils.jsonio import read_json

    path = Path(config["paths"]["manifests"]) / "g2_path_manifest.json"
    manifest = read_json(path)
    mutate(manifest)
    write_json(path, manifest)


def _rewrite_summary(config: dict[str, Any], mutate: Any) -> None:
    from pes2ts_core.utils.jsonio import read_json
    from pes2ts_core.utils.parquet_io import read_parquet

    path = Path(config["paths"]["interim"]) / "g2_summary.parquet"
    rows = [mutate(dict(row)) for row in read_parquet(path).to_pylist()]
    write_summary(path, rows)


def test_clean_tree_reports_no_problems(tmp_path: Path) -> None:
    config, _ = _build_clean_tree(tmp_path)
    outcome = verify_g2(config=config)
    assert isinstance(outcome, G2Verification)
    assert outcome.problems == ()
    assert (outcome.n_total, outcome.n_valid, outcome.n_failed) == (1, 1, 0)


def test_failed_reaction_without_frames_parquet_is_consistent(tmp_path: Path) -> None:
    config, _ = _build_clean_tree(tmp_path, failed=True)
    outcome = verify_g2(config=config)
    assert outcome.problems == ()
    assert (outcome.n_total, outcome.n_valid, outcome.n_failed) == (1, 0, 1)


def test_edited_r_xyz_digest_is_caught(tmp_path: Path) -> None:
    config, rxn_dir = _build_clean_tree(tmp_path)
    (rxn_dir / "R.xyz").write_text("2\nTAMPERED R\nC 9.9 0.0 0.0\nO 1.4 0.0 0.0\n", encoding="utf-8")
    problems = verify_g2(config=config).problems
    assert any("R.xyz" in problem and "digest" in problem for problem in problems)


def test_missing_frame_row_is_caught(tmp_path: Path) -> None:
    config, rxn_dir = _build_clean_tree(tmp_path)
    write_frames_parquet(rxn_dir / "frames.parquet", [_frame_row(REACTION_ID, 0)])
    problems = verify_g2(config=config).problems
    assert any("frames.parquet" in problem and "n_frames" in problem for problem in problems)


def test_frame_row_reaction_id_mismatch_is_caught(tmp_path: Path) -> None:
    config, rxn_dir = _build_clean_tree(tmp_path)
    rows = [_frame_row(REACTION_ID, 0), _frame_row("RXN_0000000999", 1)]
    write_frames_parquet(rxn_dir / "frames.parquet", rows)
    problems = verify_g2(config=config).problems
    assert any("reaction_id" in problem for problem in problems)


@pytest.mark.parametrize("nested", [False, True], ids=["top_level", "nested"])
def test_forbidden_coordinates_key_is_caught(tmp_path: Path, nested: bool) -> None:
    config, rxn_dir = _build_clean_tree(tmp_path)

    def mutate(document: dict[str, Any]) -> None:
        if nested:
            document["endpoints"]["coordinates"] = [[0.0, 0.0, 0.0]]
        else:
            document["coordinates"] = [[0.0, 0.0, 0.0]]

    _rewrite_document(rxn_dir, mutate)
    problems = verify_g2(config=config).problems
    assert any("'coordinates'" in problem and "reaction_path.json" in problem for problem in problems)


def test_forbidden_key_inside_endpoints_json_is_caught(tmp_path: Path) -> None:
    config, rxn_dir = _build_clean_tree(tmp_path)
    from pes2ts_core.utils.jsonio import read_json

    path = rxn_dir / "endpoints.json"
    record = read_json(path)
    record["product_atoms"] = [
        {"map": 5, "element": "O", "component": "P:P1", "local_index": 0,
         "global_index": 1, "coordinate": [2.4, 0.0, 0.0], "forces": [0.0, 0.0, 0.0]},
    ]
    write_json(path, record)
    problems = verify_g2(config=config).problems
    assert any("'forces'" in problem and "endpoints.json" in problem for problem in problems)


def test_deleted_frames_parquet_is_caught(tmp_path: Path) -> None:
    config, rxn_dir = _build_clean_tree(tmp_path)
    (rxn_dir / "frames.parquet").unlink()
    problems = verify_g2(config=config).problems
    assert any(str(rxn_dir / "frames.parquet") in problem for problem in problems)


def test_stray_document_without_summary_row_is_caught(tmp_path: Path) -> None:
    config, _ = _build_clean_tree(tmp_path)
    interim = Path(config["paths"]["interim"])
    stray_dir = interim / "g2" / "paths" / "00000" / STRAY_ID
    stray_dir.mkdir(parents=True)
    write_json(
        stray_dir / "reaction_path.json",
        {"schema_version": "g2_path_v1", "reaction_id": STRAY_ID, "status": "valid"},
    )
    problems = verify_g2(config=config).problems
    assert any(STRAY_ID in problem and "stray" in problem for problem in problems)


def test_manifest_count_tamper_is_caught(tmp_path: Path) -> None:
    config, _ = _build_clean_tree(tmp_path)
    _rewrite_manifest(config, lambda manifest: manifest.update(n_valid=manifest["n_valid"] + 1))
    problems = verify_g2(config=config).problems
    assert any("n_valid" in problem for problem in problems)


def test_manifest_summary_sha256_mismatch_is_caught(tmp_path: Path) -> None:
    config, _ = _build_clean_tree(tmp_path)
    _rewrite_manifest(config, lambda manifest: manifest.update(summary_sha256="0" * 64))
    problems = verify_g2(config=config).problems
    assert any("summary_sha256" in problem for problem in problems)


def test_manifest_missing_run_key_is_caught(tmp_path: Path) -> None:
    config, _ = _build_clean_tree(tmp_path)
    _rewrite_manifest(config, lambda manifest: manifest["run"].pop("n_skipped"))
    problems = verify_g2(config=config).problems
    assert any("n_skipped" in problem and "run" in problem for problem in problems)


def test_manifest_by_code_tamper_is_caught(tmp_path: Path) -> None:
    config, rxn_dir = _build_clean_tree(tmp_path)

    def fail_document(document: dict[str, Any]) -> None:
        document["status"] = "failed"
        document["failure_code"] = "G2_XTB_FAILED"

    _rewrite_document(rxn_dir, fail_document)
    _rewrite_summary(config, lambda row: {**row, "status": "failed", "failure_code": "G2_XTB_FAILED"})
    _rewrite_manifest(config, lambda manifest: manifest.update(by_code={}))
    problems = verify_g2(config=config).problems
    assert any("by_code" in problem for problem in problems)


def test_summary_row_document_status_mismatch_is_caught(tmp_path: Path) -> None:
    config, _ = _build_clean_tree(tmp_path)
    _rewrite_summary(config, lambda row: {**row, "status": "failed", "failure_code": "G2_XTB_FAILED"})
    problems = verify_g2(config=config).problems
    assert any(REACTION_ID in problem and "status" in problem for problem in problems)


def test_schema_version_mismatch_is_caught(tmp_path: Path) -> None:
    config, rxn_dir = _build_clean_tree(tmp_path)
    _rewrite_document(rxn_dir, lambda document: document.update(schema_version="g2_path_v0"))
    problems = verify_g2(config=config).problems
    assert any("schema_version" in problem for problem in problems)


def test_problem_order_is_deterministic_across_calls(tmp_path: Path) -> None:
    config, rxn_dir = _build_clean_tree(tmp_path)
    (rxn_dir / "R.xyz").write_text("2\nTAMPERED R\nC 9.9 0.0 0.0\nO 1.4 0.0 0.0\n", encoding="utf-8")
    (rxn_dir / "frames.parquet").unlink()
    _rewrite_summary(config, lambda row: {**row, "status": "failed"})
    first = verify_g2(config=config).problems
    second = verify_g2(config=config).problems
    assert first == second
    assert len(first) >= 3


def test_missing_summary_raises_file_not_found(tmp_path: Path) -> None:
    config = _config(tmp_path)
    with pytest.raises(FileNotFoundError):
        verify_g2(config=config)


def test_expected_digest_of_stable_rows_matches_manifest(tmp_path: Path) -> None:
    """Sanity: the fixture manifest digest is the canonical rows digest."""
    config, _ = _build_clean_tree(tmp_path)
    from pes2ts_core.utils.jsonio import read_json
    from pes2ts_core.utils.parquet_io import read_parquet

    rows = read_parquet(Path(config["paths"]["interim"]) / "g2_summary.parquet").to_pylist()
    manifest = read_json(Path(config["paths"]["manifests"]) / "g2_path_manifest.json")
    assert manifest["summary_sha256"] == sha256_bytes(stable_json_dumps(rows).encode("utf-8"))
