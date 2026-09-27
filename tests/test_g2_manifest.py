"""TDD tests for ``pes2ts_core.g2.artifacts`` (task 8).

Covers the per-reaction path document, the frames parquet, the batch summary
parquet, the ``g2_manifest_v1`` manifest (injectable ``generated_at``, ``run``
count block, g2-specific ``config_digest``, count-mismatch guard, empty batch),
and the ``g2_coverage_v1`` category cross — plus the byte-determinism contract
(strip ``generated_at`` and the ``run`` block; parquet bytes stable).
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Final

import pytest
from pes2ts_core.g1.document import CATEGORY_NAMES
from pes2ts_core.g2 import SCHEMA_COVERAGE, SCHEMA_MANIFEST, SCHEMA_PATH, artifacts
from pes2ts_core.utils.hashing import sha256_bytes, stable_json_dumps
from pes2ts_core.utils.jsonio import read_json
from pes2ts_core.utils.parquet_io import read_parquet

CONFIG: Final[dict[str, Any]] = {
    "g2": {
        "validity": {"endpoint_rmsd_max": 0.5, "min_frames": 8},
        "xtb": {"seed": 42, "threads": 4},
    },
}
CONFIG_ALT_THRESHOLD: Final[dict[str, Any]] = {
    "g2": {
        "validity": {"endpoint_rmsd_max": 0.9, "min_frames": 8},
        "xtb": {"seed": 42, "threads": 4},
    },
}
XTB_FINGERPRINT: Final[dict[str, Any]] = {
    "sha256": "e" * 64,
    "version": "6.7.1",
    "path_inp": "$path nrun=1 npoint=25 $end",
    "seed_supported": True,
}
RUN_COUNTS: Final[dict[str, int]] = {"n_selected": 2, "n_attempted": 2, "n_skipped": 0}
GENERATED_AT: Final[str] = "2026-01-01T00:00:00+00:00"


def _frame_row(reaction_id: str, index: int, energy: float | None) -> dict[str, Any]:
    return {
        "reaction_id": reaction_id,
        "frame_index": index,
        "energy_rel_kcal": energy,
        # Forward rows keep raw == energy_rel_kcal (task 9 interface extension).
        "energy_rel_kcal_raw": energy,
        "rmsd_to_start": 0.1 * index,
        "rmsd_to_end": 1.0 - 0.1 * index,
        "step_max": 0.05,
        "step_rmsd": 0.02,
        "min_nonbonded_distance": 1.2 + 0.01 * index,
        "event_distances": {"formed:1-2": 2.5 + 0.1 * index},
    }


def _frames(reaction_id: str = "RXN_A") -> list[dict[str, Any]]:
    return [_frame_row(reaction_id, index, float(index)) for index in range(3)]


def _document(
    reaction_id: str = "RXN_A",
    *,
    status: str = "valid",
    failure_code: str | None = None,
    direction: str = "forward",
    direction_recovered: bool = False,
) -> dict[str, Any]:
    return artifacts.reaction_document(
        reaction_id=reaction_id,
        status=status,
        failure_code=failure_code,
        failure_detail="detail" if failure_code else None,
        direction=direction,
        direction_recovered=direction_recovered,
        dataset_version="zenodo-18551029-rev1",
        sources={"g1_document_sha256": "a" * 64, "R_xyz_sha256": "b" * 64},
        endpoints={
            "anchor_component": "R:0",
            "global_frame_basis": "anchor_shared_maps",
            "multiplicity_basis": "g1_valid_invariant",
        },
        attempts=[
            {
                "direction": "forward",
                "argv": ["xtb", "start.xyz", "--path", "end.xyz"],
                "returncode": 0,
                "timed_out": False,
                "duration_seconds": 1.5,
                "xtb_sha256": "c" * 64,
            }
        ],
        frames=_frames(reaction_id),
        validity={"n_frames": 3, "first_vs_r_rmsd": 0.1, "last_vs_p_rmsd": 0.1},
        config_digest="d" * 64,
    )


def _strip_volatile(manifest: Mapping[str, Any]) -> str:
    """Return the canonical JSON of a manifest minus generated_at and run."""
    stripped = {key: value for key, value in manifest.items() if key not in {"generated_at", "run"}}
    return stable_json_dumps(stripped)


def _walk_keys(value: Any) -> Iterator[str]:
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield str(key)
            yield from _walk_keys(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _walk_keys(item)


# --- reaction_document -------------------------------------------------------


def test_reaction_document_shape_when_built_from_pipeline_inputs() -> None:
    document = _document()
    assert document["schema_version"] == SCHEMA_PATH
    assert document["reaction_id"] == "RXN_A"
    assert document["dataset_version"] == "zenodo-18551029-rev1"
    assert document["status"] == "valid"
    assert document["failure_code"] is None
    assert document["direction"] == "forward"
    assert document["direction_recovered"] is False
    assert document["sources"]["g1_document_sha256"] == "a" * 64
    assert document["endpoints"]["anchor_component"] == "R:0"
    assert document["endpoints"]["multiplicity_basis"] == "g1_valid_invariant"
    assert document["attempts"][0]["returncode"] == 0
    assert document["config_digest"] == "d" * 64


def test_reaction_document_frames_summary_when_frames_supplied() -> None:
    summary = _document()["frames"]
    assert summary["n_frames"] == 3
    assert summary["energy_min"] == 0.0
    assert summary["energy_max"] == 2.0
    assert summary["worst_step_max"] == 0.05
    assert summary["min_nonbonded"] == 1.2


def test_reaction_document_failure_fields_when_status_failed() -> None:
    document = _document(status="failed", failure_code="G2_COLLISION", direction="reverse")
    assert document["status"] == "failed"
    assert document["failure_code"] == "G2_COLLISION"
    assert document["failure_detail"] == "detail"
    assert document["direction"] == "reverse"


def test_reaction_document_no_forbidden_keys_or_arrays_when_inspected() -> None:
    document = _document()
    keys = set(_walk_keys(document))
    assert not keys & {"coordinates", "EHG", "forces"}
    frames_block = document["frames"]
    assert all(not isinstance(value, (list, tuple)) for value in frames_block.values())


def test_reaction_document_is_deterministic_when_built_twice() -> None:
    assert stable_json_dumps(_document()) == stable_json_dumps(_document())


# --- frames parquet ----------------------------------------------------------


def test_write_frames_parquet_roundtrip_and_column_order(tmp_path: Path) -> None:
    path = tmp_path / "frames.parquet"
    artifacts.write_frames_parquet(path, _frames())
    table = read_parquet(path)
    assert list(table.column_names) == sorted(artifacts.FRAME_COLUMNS)
    rows = table.to_pylist()
    assert [row["frame_index"] for row in rows] == [0, 1, 2]
    assert rows[1]["energy_rel_kcal"] == 1.0
    assert rows[1]["energy_rel_kcal_raw"] == 1.0
    assert json.loads(rows[0]["event_distances"]) == {"formed:1-2": 2.5}
    assert rows[2]["min_nonbonded_distance"] == pytest.approx(1.22)


def test_write_frames_parquet_bytes_stable_when_written_twice(tmp_path: Path) -> None:
    first, second = tmp_path / "a.parquet", tmp_path / "b.parquet"
    artifacts.write_frames_parquet(first, _frames())
    artifacts.write_frames_parquet(second, _frames())
    assert first.read_bytes() == second.read_bytes()


def test_write_frames_parquet_null_energy_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "frames.parquet"
    artifacts.write_frames_parquet(path, [_frame_row("RXN_A", 0, None)])
    rows = read_parquet(path).to_pylist()
    assert rows[0]["energy_rel_kcal"] is None


# --- summary -----------------------------------------------------------------


def test_build_summary_rows_when_documents_supplied() -> None:
    documents = [
        _document("RXN_A"),
        _document("RXN_B", status="failed", failure_code="G2_PATH_DISCONTINUOUS",
                  direction="reverse", direction_recovered=False),
    ]
    rows = artifacts.build_summary(documents)
    assert [row["reaction_id"] for row in rows] == ["RXN_A", "RXN_B"]
    assert rows[0]["status"] == "valid"
    assert rows[0]["n_frames"] == 3
    assert rows[0]["n_attempts"] == 1
    assert rows[1]["failure_code"] == "G2_PATH_DISCONTINUOUS"
    assert rows[1]["direction"] == "reverse"
    assert rows[1]["n_frames"] == 3
    assert rows[1]["n_attempts"] == 1


def test_write_summary_roundtrip_and_bytes_stable(tmp_path: Path) -> None:
    documents = [_document("RXN_A"), _document("RXN_B", status="failed", failure_code="G2_COLLISION")]
    first, second = tmp_path / "a.parquet", tmp_path / "b.parquet"
    artifacts.write_summary(first, artifacts.build_summary(documents))
    artifacts.write_summary(second, artifacts.build_summary(documents))
    assert list(read_parquet(first).column_names) == sorted(artifacts.SUMMARY_COLUMNS)
    assert first.read_bytes() == second.read_bytes()


# --- manifest ----------------------------------------------------------------


def _summary_rows() -> list[dict[str, Any]]:
    return artifacts.build_summary(
        [
            _document("RXN_A"),
            _document("RXN_B", status="failed", failure_code="G2_COLLISION",
                      direction="reverse", direction_recovered=True),
        ]
    )


def test_manifest_fields_when_built_from_summary_rows() -> None:
    manifest = artifacts.build_manifest(
        _summary_rows(), run_counts=RUN_COUNTS, xtb_fingerprint=XTB_FINGERPRINT,
        config=CONFIG, generated_at=GENERATED_AT,
    )
    assert manifest["schema_version"] == SCHEMA_MANIFEST
    assert manifest["n_total"] == 2
    assert manifest["n_valid"] == 1
    assert manifest["n_failed"] == 1
    assert manifest["by_code"] == {"G2_COLLISION": 1}
    assert manifest["reverse_recovery_rate"] == 0.5
    assert manifest["run"] == RUN_COUNTS
    assert manifest["xtb"] == XTB_FINGERPRINT
    assert manifest["generated_at"] == GENERATED_AT
    rows = _summary_rows()
    assert manifest["summary_sha256"] == sha256_bytes(stable_json_dumps(rows).encode("utf-8"))
    assert manifest["config_digest"] == artifacts.config_digest(
        CONFIG, xtb_sha256=XTB_FINGERPRINT["sha256"], path_inp=XTB_FINGERPRINT["path_inp"]
    )


def test_manifest_is_byte_identical_when_volatile_fields_stripped(tmp_path: Path) -> None:
    first = artifacts.write_manifest(
        tmp_path / "a.json", _summary_rows(), run_counts=RUN_COUNTS,
        xtb_fingerprint=XTB_FINGERPRINT, config=CONFIG, generated_at=GENERATED_AT,
    )
    second = artifacts.write_manifest(
        tmp_path / "b.json", _summary_rows(), run_counts={"n_selected": 9, "n_attempted": 1, "n_skipped": 8},
        xtb_fingerprint=XTB_FINGERPRINT, config=CONFIG, generated_at="2027-12-31T23:59:59+00:00",
    )
    assert _strip_volatile(first) == _strip_volatile(second)
    assert read_json(tmp_path / "a.json")["schema_version"] == SCHEMA_MANIFEST


def test_manifest_config_digest_changes_when_threshold_changes() -> None:
    base = artifacts.config_digest(
        CONFIG, xtb_sha256=XTB_FINGERPRINT["sha256"], path_inp=XTB_FINGERPRINT["path_inp"]
    )
    altered = artifacts.config_digest(
        CONFIG_ALT_THRESHOLD, xtb_sha256=XTB_FINGERPRINT["sha256"], path_inp=XTB_FINGERPRINT["path_inp"]
    )
    assert base != altered


def test_manifest_count_mismatch_raises_when_override_contradicts_rows() -> None:
    with pytest.raises(artifacts.ManifestCountMismatch):
        artifacts.build_manifest(
            _summary_rows(), run_counts=RUN_COUNTS, xtb_fingerprint=XTB_FINGERPRINT,
            config=CONFIG, generated_at=GENERATED_AT,
            counts={"n_total": 5, "n_valid": 5, "n_failed": 0},
        )


def test_manifest_when_batch_empty(tmp_path: Path) -> None:
    manifest = artifacts.write_manifest(
        tmp_path / "manifest.json", [], run_counts={"n_selected": 0, "n_attempted": 0, "n_skipped": 0},
        xtb_fingerprint=XTB_FINGERPRINT, config=CONFIG, generated_at=GENERATED_AT,
    )
    assert manifest["schema_version"] == SCHEMA_MANIFEST
    assert manifest["n_total"] == 0
    assert manifest["n_valid"] == 0
    assert manifest["n_failed"] == 0
    assert manifest["by_code"] == {}
    assert manifest["reverse_recovery_rate"] == 0.0


# --- coverage ----------------------------------------------------------------


def test_coverage_cross_and_history_when_written(tmp_path: Path) -> None:
    categories_by_reaction = {
        "RXN_A": dict.fromkeys(CATEGORY_NAMES, False) | {"pure_formed": True},
        "RXN_B": dict.fromkeys(CATEGORY_NAMES, False) | {"multi_component": True, "both": True},
    }
    coverage = artifacts.write_coverage(
        tmp_path / "coverage.json",
        _summary_rows(),
        categories_by_reaction=categories_by_reaction,
        current_by_code={"G2_COLLISION": 1},
        historical_failed_by_code={"G2_ENDPOINT_NOT_REACHED": 3},
        generated_at=GENERATED_AT,
    )
    assert coverage["schema_version"] == SCHEMA_COVERAGE
    assert sorted(coverage["categories"]) == sorted(CATEGORY_NAMES)
    assert coverage["categories"]["pure_formed"] == {"n_total": 1, "n_valid": 1, "n_failed": 0}
    assert coverage["categories"]["multi_component"] == {"n_total": 1, "n_valid": 0, "n_failed": 1}
    assert coverage["categories"]["h_migration"] == {"n_total": 0, "n_valid": 0, "n_failed": 0}
    assert coverage["current_by_code"] == {"G2_COLLISION": 1}
    assert coverage["historical_failed_by_code"] == {"G2_ENDPOINT_NOT_REACHED": 3}
    assert coverage["generated_at"] == GENERATED_AT
    assert read_json(tmp_path / "coverage.json") == coverage


def test_coverage_is_deterministic_when_volatile_stripped(tmp_path: Path) -> None:
    kwargs: dict[str, Any] = {
        "categories_by_reaction": {"RXN_A": dict.fromkeys(CATEGORY_NAMES, False)},
        "current_by_code": {},
        "historical_failed_by_code": {},
    }
    first = artifacts.write_coverage(tmp_path / "a.json", _summary_rows(), generated_at="2026-01-01T00:00:00+00:00", **kwargs)
    second = artifacts.write_coverage(tmp_path / "b.json", _summary_rows(), generated_at="2027-01-01T00:00:00+00:00", **kwargs)
    assert stable_json_dumps({k: v for k, v in first.items() if k != "generated_at"}) == stable_json_dumps(
        {k: v for k, v in second.items() if k != "generated_at"}
    )
