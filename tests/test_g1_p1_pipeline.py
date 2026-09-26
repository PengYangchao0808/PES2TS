"""End-to-end P1/P2 pipeline on a synthetic truth fixture.

The fixture builds one dissociation reaction (contiguous maps 1..6) through
the real ``build_g1`` path, then fabricates a quarantined truth tree -- a
map-ordered ``ts.parquet``, an IRC HDF5 in the documented
``[TS, branch->R, branch->P]`` layout behind a real ``truth_manifest.json``
-- and drives ``resolve_p1`` -> ``classify_p2`` -> ``verify_p1`` ->
``verify_p2`` -> ``run_gate`` plus the CLI contracts (``--allow-truth``
gating, exit codes, audited access log).  A second reaction absent from the
IRC archive exercises the typed ``missing_truth_join`` path end to end.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pyarrow as pa
import pytest

from pes2ts_core.cli import main
from pes2ts_core.g1.build import build_g1
from pes2ts_core.g1.gate import run_gate
from pes2ts_core.g1.p1_truth import (
    P1_SUMMARY_COLUMNS,
    p1_document_path,
    resolve_p1,
    run_join_audit,
)
from pes2ts_core.g1.p1_verify import verify_p1
from pes2ts_core.g1.p2_build import classify_p2
from pes2ts_core.g1.p2_verify import verify_p2
from pes2ts_core.g1.truth_schema import (
    EXIT_TRUTH_FLAG_REQUIRED,
    JOIN_AUDIT_FILENAME,
    P1_SUMMARY_FILENAME,
    STATUS_MISSING_TRUTH_JOIN,
    STATUS_RESOLVED_UNIQUE,
)
from pes2ts_core.utils.hashing import sha256_file
from pes2ts_core.utils.jsonio import read_json, write_json
from pes2ts_core.utils.parquet_io import read_parquet, write_parquet

RXN_OK = "RXN_0000000001"
RXN_NO_IRC = "RXN_0000000002"
REACTION_SMILES = "[F:1][C:2]([H:3])([Cl:4])[O:5][H:6]>>[F:1][C:2]([H:3])([Cl:4])[O:5].[H:6]"
R0_SMILES = "[F:1][C:2]([H:3])([Cl:4])[O:5][H:6]"
P0_SMILES = "[F:1][C:2]([H:3])([Cl:4])[O:5]"
P1_SMILES = "[H:1]"

R0_Z = (9, 6, 1, 17, 8, 1)
R0_X = np.array([
    [0.0, 0.0, 0.0], [1.35, 0.0, 0.0], [1.9, 0.6, 0.3],
    [1.7, -1.2, 0.6], [2.6, 0.7, -0.2], [3.55, 1.2, 0.12],
])
P0_Z = (9, 6, 1, 17, 8)
P0_X = R0_X[:5]
P1_Z = (1,)
P1_X = np.array([[4.6, 1.9, 0.4]])
TS_X = R0_X.copy()
TS_X[5] = R0_X[4] + (R0_X[5] - R0_X[4]) / np.linalg.norm(R0_X[5] - R0_X[4]) * 1.6


def _component(tag: str, smiles: str, numbers: tuple[int, ...], coordinates: np.ndarray) -> dict[str, Any]:
    return {
        "tag": tag, "smiles": smiles, "atomic_numbers": list(numbers),
        "coordinates": [list(map(float, row)) for row in coordinates],
        "charge": 0, "multiplicity": 1, "EHG": [-1.0, -1.0, -1.0],
    }


def _inventory_row(reaction_id: str) -> dict[str, Any]:
    return {
        "reaction_id": reaction_id, "dataset_version": "zenodo-1-rev1",
        "reaction_smiles": REACTION_SMILES,
        "n_reactant_components": 1, "n_product_components": 2,
        "elements": ["F", "C", "H", "Cl", "O", "H"],
        "total_atoms_reactants": 6, "total_atoms_products": 6,
        "charge_total_reactants": 0, "charge_total_products": 0,
        "multiplicity_max": 1,
        "components": [
            _component("R0", R0_SMILES, R0_Z, R0_X),
            _component("P0", P0_SMILES, P0_Z, P0_X),
            _component("P1", P1_SMILES, P1_Z, P1_X),
        ],
    }


def _config(root: Path) -> dict[str, Any]:
    data_root = root / "data"
    return {
        "source": {"zenodo_record": 1, "zenodo_revision": "1", "files": {}},
        "paths": {
            "data_root": str(data_root),
            "raw": str(data_root / "raw"),
            "interim": str(data_root / "interim"),
            "ground_truth": str(data_root / "ground_truth"),
            "truth_sources": str(data_root / "ground_truth" / "sources"),
            "manifests": str(data_root / "manifests"),
        },
        "split": {"seed": 42},
        "g1": {
            "shard_size": 1000, "match_cap": 10000, "max_candidates": 64,
            "bond_tolerance": 0.45, "neighborhood_shell": 1,
            "sample_size": 20, "sample_seed": 42,
        },
    }


def _write_inventory(config: dict[str, Any]) -> None:
    rows = [_inventory_row(RXN_OK), _inventory_row(RXN_NO_IRC)]
    fields = (
        "reaction_id", "dataset_version", "reaction_smiles",
        "n_reactant_components", "n_product_components", "elements",
        "total_atoms_reactants", "total_atoms_products",
        "charge_total_reactants", "charge_total_products", "multiplicity_max",
        "components",
    )
    write_parquet(
        Path(config["paths"]["interim"]) / "inventory.parquet",
        pa.table({field: [row[field] for row in rows] for field in fields}),
    )


def _write_truth_tree(config: dict[str, Any]) -> None:
    """Write ts.parquet, the IRC archive, and the truth manifest."""
    ground_truth = Path(config["paths"]["ground_truth"])
    sources = Path(config["paths"]["truth_sources"])
    sources.mkdir(parents=True, exist_ok=True)
    ts_rows = {
        "reaction_id": [RXN_OK, RXN_NO_IRC],
        "atomic_numbers": [list(R0_Z), list(R0_Z)],
        "coordinates": [
            [list(map(float, row)) for row in TS_X],
            [list(map(float, row)) for row in TS_X],
        ],
        "EHG": [[-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
        "charge": [0, 0], "multiplicity": [1, 1],
        "reaction_smiles": [REACTION_SMILES, REACTION_SMILES],
    }
    ts_path = ground_truth / "ts.parquet"
    write_parquet(ts_path, pa.table(ts_rows))
    irc_index_rows = {
        "reaction_id": [RXN_OK], "n_atoms": [6], "n_frames": [17],
        "has_forces": [False], "ts_index": [0],
    }
    irc_index_path = ground_truth / "irc_index.parquet"
    write_parquet(irc_index_path, pa.table(irc_index_rows))
    irc_h5_path = sources / "fixture_TZVP_IRC.h5"
    reactant = R0_X
    product = np.vstack([P0_X, P1_X])
    branch_a = np.linspace(0.0, 1.0, 9)[:, None, None] * (reactant - TS_X) + TS_X
    branch_b = np.linspace(0.1, 1.0, 8)[:, None, None] * (product - TS_X) + TS_X
    frames = np.vstack([branch_a, branch_b])
    with h5py.File(irc_h5_path, "w") as handle:
        bundle = handle.create_group("bundle_a")
        group = bundle.create_group(RXN_OK)
        group.create_dataset("coordinates", data=frames)
    write_json(Path(config["paths"]["manifests"]) / "truth_manifest.json", {
        "schema_version": "g0_manifest_v1",
        "dataset_version": "zenodo-1-rev1",
        "ts_parquet": {"path": str(ts_path), "sha256": sha256_file(ts_path), "n_rows": 2},
        "irc_index": {"path": str(irc_index_path), "sha256": sha256_file(irc_index_path), "n_rows": 1},
        "sources": [{
            "filename": irc_h5_path.name,
            "original_path": str(irc_h5_path),
            "relocated_path": str(irc_h5_path),
            "sha256": sha256_file(irc_h5_path),
            "size_bytes": irc_h5_path.stat().st_size,
        }],
        "n_reactions_without_ts": 0,
    })


def _build_fixture(root: Path) -> dict[str, Any]:
    config = _config(root)
    Path(config["paths"]["interim"]).mkdir(parents=True, exist_ok=True)
    _write_inventory(config)
    _write_truth_tree(config)
    build_g1(config)
    return config


def test_full_p1_p2_pipeline_on_synthetic_truth(tmp_path: Path) -> None:
    config = _build_fixture(tmp_path)
    # P1.0: the join audit explains both reactions (one lacks the IRC join).
    audit_path = run_join_audit(config, allow_truth=True)
    audit = read_json(audit_path)
    assert audit["n_joined_all"] == 1
    assert audit["by_reason"].get("missing_irc") == 1
    # P1 resolve: one resolved reaction with a complete map table.
    result = resolve_p1(config, allow_truth=True)
    assert result.n_total == 2
    assert result.by_status.get(STATUS_RESOLVED_UNIQUE) == 1
    assert result.by_status.get(STATUS_MISSING_TRUTH_JOIN) == 1
    document = read_json(p1_document_path(
        Path(config["paths"]["interim"]), RXN_OK, 1000,
    ))
    assert document["status"] == STATUS_RESOLVED_UNIQUE
    assert document["truth_assisted"] is True
    maps = [entry["map"] for entry in document["mapping"]["map_to_atoms"]]
    assert maps == [1, 2, 3, 4, 5, 6]
    assert [entry["ts_irc_index"] for entry in document["mapping"]["map_to_atoms"]] == [0, 1, 2, 3, 4, 5]
    assert document["irc_validation"]["endpoint_match"] == "pass"
    assert document["irc_validation"]["orientation"] == "R_first"
    assert document["irc_validation"]["n_support"] >= 1
    broken = document["bond_events"]["broken"]
    assert broken == [{"atoms": [5, 6], "order_r": 1.0, "order_p": None}]
    summary = read_parquet(
        Path(config["paths"]["interim"]) / P1_SUMMARY_FILENAME
    ).to_pylist()
    assert [row["reaction_id"] for row in summary] == [RXN_OK, RXN_NO_IRC]
    assert summary[0]["g2_eligible"] is True
    assert summary[1]["g2_eligible"] is False
    # P2 classify: one classified reaction, one excluded by its P1 status.
    p2 = classify_p2(config)
    assert p2.n_classified == 1
    assert p2.n_excluded == 1
    # Verification and gate reconcile everything.
    assert verify_p1(config).problems == ()
    assert verify_p2(config).problems == ()
    gate = run_gate(config)
    assert gate.refusals == ()
    assert gate.n_denominator == 2
    assert gate.n_eligible == 1
    assert gate.eligible_fraction == 0.5
    eligible = read_json(gate.eligible_path)
    assert eligible["reaction_ids"] == [RXN_OK]
    # Every audited truth read is in the access log with the P1 caller.
    log_path = Path(config["paths"]["manifests"]) / "truth_access_log.jsonl"
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert any(entry["function"] == "load_ts_table" for entry in lines)
    assert any(entry["function"] == "load_irc_frames" and entry.get("caller") == "pes2ts_core.g1.p1_truth" for entry in lines)


def test_resolve_refuses_without_allow_truth(tmp_path: Path) -> None:
    config = _build_fixture(tmp_path)
    with pytest.raises(PermissionError):
        resolve_p1(config, allow_truth=False)


def test_cli_contracts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _build_fixture(tmp_path)
    config_path = tmp_path / "fixture.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert main(["--config", str(config_path), "g1", "resolve-map", "--allow-truth"]) == 0
    assert main(["--config", str(config_path), "g1", "resolve-map"]) == EXIT_TRUTH_FLAG_REQUIRED
    assert main(["--config", str(config_path), "g1", "join-audit"]) == EXIT_TRUTH_FLAG_REQUIRED
    assert main(["--config", str(config_path), "g1", "join-audit", "--allow-truth"]) == 0
    assert main(["--config", str(config_path), "g1", "classify"]) == 0
    assert main(["--config", str(config_path), "g1", "verify", "--stage", "p1"]) == 0
    assert main(["--config", str(config_path), "g1", "verify", "--stage", "p2"]) == 0
    assert main(["--config", str(config_path), "g1", "verify", "--stage", "build"]) == 0
    assert main(["--config", str(config_path), "g1", "gate"]) == 0
    manifests = Path(config["paths"]["manifests"])
    assert (manifests / JOIN_AUDIT_FILENAME).is_file()
    assert (manifests / "g1_p1_manifest.json").is_file()
    assert (manifests / "g1_p2_manifest.json").is_file()
    assert (manifests / "g1_cluster_report.json").is_file()
    assert (manifests / "g1_gate.json").is_file()
    assert (Path(config["paths"]["interim"]) / "g2_eligible.json").is_file()


def test_p1_documents_are_deterministic(tmp_path: Path) -> None:
    config = _build_fixture(tmp_path)
    first = resolve_p1(config, allow_truth=True)
    interim = Path(config["paths"]["interim"])
    first_document = read_json(p1_document_path(interim, RXN_OK, 1000))
    first_summary = sha256_file(interim / P1_SUMMARY_FILENAME)
    second = resolve_p1(config, allow_truth=True)
    assert second.n_eligible == first.n_eligible
    second_document = read_json(p1_document_path(interim, RXN_OK, 1000))
    second_summary = sha256_file(interim / P1_SUMMARY_FILENAME)
    for document in (first_document, second_document):
        document.pop("generated_at", None)
    assert first_document == second_document
    assert first_summary == second_summary
