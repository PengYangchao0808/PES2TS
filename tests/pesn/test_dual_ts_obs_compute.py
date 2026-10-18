"""Synthetic tests for the WP2-B resumable E/g observation driver.

No ACP subprocess, no xTB, no network: the compute unit is injected as a
fake that writes receipts through the REAL frozen request builder
(``build_gradient_request`` + ``digest``) and the backend's atomic
``write_json``, so receipt validation exercises the same binding rule the
real backend receipts satisfy.  Receipt layout, resume semantics, failure
registry, ledger accounting and unit annotation are all checked on disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pes2ts_core.generation.planning.synchronized_path import digest  # noqa: E402
from pes2ts_core.integration.acp.gradient_backend import write_json  # noqa: E402
from pes2ts_core.integration.acp.orca_gradient_request import (  # noqa: E402
    build_gradient_request,
)
from pes2ts_core.pesn.offline.data.dual_ts_fit50 import (  # noqa: E402
    atom_order_hash,
    computation_identity,
    geometry_content_digest,
)
from scripts.compute_dual_ts_observations import (  # noqa: E402
    CoverageBindingError,
    annotate_receipt,
    build_node_payloads,
    parse_args,
    receipt_dir,
    receipt_state,
    run_driver,
    settle_failed_receipt,
    units_annotation,
    validate_receipt,
)

XTB_BLOCKS = ['%xtb\n XTBINPUTSTRING "--iterations 2000"\nend']

Z = [1, 8, 1]  # 3-atom synthetic molecule


def _coords(seed: int) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    return [[round(float(v), 10) for v in row]
            for row in rng.normal(scale=0.5, size=(3, 3))]


def _node(reaction_id: str, coords, cache_status="to_compute",
          refs=None, is_ts=False) -> dict:
    geom = geometry_content_digest(np.asarray(coords, dtype=float))
    order = atom_order_hash(np.asarray(Z, dtype=int),
                            [f"m{i + 1:04d}" for i in range(len(Z))])
    identity = computation_identity(geom, order, "GFN2-xTB", 0, 1,
                                    "eg_single_point")
    return {
        "node_id": f"{reaction_id}|node|{identity['digest'][:12]}",
        "reaction_id": reaction_id,
        "reaction_group_id": reaction_id,
        "geometry_content_digest": geom,
        "atom_order_hash": order,
        "method": "GFN2-xTB",
        "charge": 0,
        "multiplicity": 1,
        "operation": "eg_single_point",
        "computation_identity": identity["digest"],
        "state_refs": refs or [{"state_id": f"{reaction_id}|A|0000",
                                "branch_id": f"{reaction_id}#brA",
                                "step_index": 0, "is_terminal": False,
                                "inserted": False}],
        "is_ts": is_ts,
        "inserted": False,
        "inserted_state_ids": [],
        "cache": {"status": cache_status, "receipt_path": None,
                  "request_sha256": None, "duration_seconds": None,
                  "n_receipts": 0},
    }


@pytest.fixture()
def coverage_dir(tmp_path: Path) -> Path:
    """One synthetic reaction: states A0, A1, TS(A terminal), B0, B1(=TS).

    Unique nodes: A0, A1, TS (shared, cache_hit), B0 -> 4 nodes, 3 to_compute.
    """
    coords = {sid: _coords(i) for i, sid in
              enumerate(["A0", "A1", "TS", "B0"])}
    states = []
    for branch, sids in (("A", ["A0", "A1", "TS"]), ("B", ["B0", "TS"])):
        for step, sid in enumerate(sids):
            full = f"RXN_T|{branch}|{step:04d}"
            states.append({
                "state_id": full, "branch_id": f"RXN_T#br{branch}",
                "step_index": step,
                "atomic_numbers": list(Z),
                "coordinates": coords[sid],
                "atom_map_ids": [f"m{i + 1:04d}" for i in range(len(Z))],
                "boundary_type": "teacher_endpoint" if sid == "TS" else None,
            })
    (tmp_path / "RXN_T.teacher.json").write_text(
        json.dumps({"schema_version": "pesn_dual_ts_teacher_v2",
                    "reaction_id": "RXN_T", "states": states}),
        encoding="utf-8")
    nodes = [
        _node("RXN_T", coords["A0"]),
        _node("RXN_T", coords["A1"],
              refs=[{"state_id": "RXN_T|A|0001",
                     "branch_id": "RXN_T#brA", "step_index": 1,
                     "is_terminal": False, "inserted": False}]),
        _node("RXN_T", coords["TS"], cache_status="cache_hit",
              refs=[{"state_id": "RXN_T|A|0002",
                     "branch_id": "RXN_T#brA", "step_index": 2,
                     "is_terminal": True, "inserted": False},
                    {"state_id": "RXN_T|B|0001",
                     "branch_id": "RXN_T#brB", "step_index": 1,
                     "is_terminal": True, "inserted": False}],
              is_ts=True),
        _node("RXN_T", coords["B0"],
              refs=[{"state_id": "RXN_T|B|0000",
                     "branch_id": "RXN_T#brB", "step_index": 0,
                     "is_terminal": False, "inserted": False}]),
    ]
    (tmp_path / "coverage_nodes.json").write_text(
        json.dumps({"schema_version": "pes2ts_dual_ts_wp2_coverage_nodes_v0",
                    "nodes": nodes}), encoding="utf-8")
    return tmp_path


def make_args(coverage_dir: Path, out_root: Path, **overrides):
    argv = ["--coverage", str(coverage_dir / "coverage_nodes.json"),
            "--out", str(out_root), "--workers", "1"]
    for key, value in overrides.items():
        flag = f"--{key.replace('_', '-')}"
        if value is True:
            argv.append(flag)
        else:
            argv += [flag, str(value)]
    args = parse_args(argv)
    args._acp_test_settings = {
        "root": "/fake/acp", "python": "/fake/python",
        "config_path": "/fake/acp.yaml", "method": "GFN2-xTB",
        "timeout_seconds": 120.0, "nproc": 2,
    }
    return args


def fake_backend_record(payload: dict, *, success: bool = True,
                        energy=..., gradient=..., request_sha="") -> dict:
    """A result.json shaped exactly like the real backend's record."""
    return {
        "success": success,
        "failure_class": None if success else
        "PROCESS_FAILED_WITHOUT_NORMAL_TERMINATION",
        "request_sha256": request_sha,
        "energy": (float(energy) if energy is not ... and energy is not None
                   else None),
        "gradient_evidence": {"status": "bound" if success else "unbound"},
        "gradient_hartree_per_angstrom": (
            gradient if gradient is not ... else None),
        "duration_seconds": 0.25,
        "method": "GFN2-xTB",
        "normal_termination": success,
    }


def write_receipt(out_root: Path, payload: dict, record: dict) -> Path:
    """Write a request+result pair like the backend (real binding digest)."""
    request = build_gradient_request(
        geometry=payload["coordinates"], elements=["H", "O", "H"],
        method=payload["method"], basis="", charge=payload["charge"],
        multiplicity=payload["multiplicity"], route_extras=["EnGrad"],
        timeout_seconds=120, nproc=2, extra_blocks=XTB_BLOCKS,
        scf_convergence="tight", output_name="grad")
    binding = digest(request)
    rd = receipt_dir(out_root, payload["digest"])
    rd.mkdir(parents=True, exist_ok=True)
    write_json(rd / "request.json",
               {**request, "request_sha256": binding,
                "acp_request_sha256": "acp-" + binding[:16]})
    if record is not None:
        record = dict(record, request_sha256=binding)
        write_json(rd / "result.json", record)
    return rd


VALID_RECORD = dict(success=True, energy=-19.0995, gradient=[[0.1, -0.2, 0.3],
                                                            [0.0, 0.05, -0.05],
                                                            [-0.1, 0.15, -0.2]])


def fake_compute_factory(mode: str = "ok", calls: list | None = None):
    """Injectable compute unit: writes backend-shaped receipts on disk."""

    def compute(payload: dict, acp: dict, out_root: str) -> dict:
        if calls is not None:
            calls.append(payload["digest"])
        settle_failed_receipt(out_root, payload["digest"])
        rd = receipt_dir(out_root, payload["digest"])
        record = fake_backend_record(
            payload, success=(mode == "ok"),
            energy=VALID_RECORD["energy"] if mode == "ok" else None,
            gradient=(VALID_RECORD["gradient"] if mode == "ok"
                      else [[0.0, 0.0, 0.0]] * payload["n_atoms"]))
        write_receipt(Path(out_root), payload, record)
        return {"digest": payload["digest"], "node_id": payload["node_id"],
                "status": "computed" if mode == "ok" else "failed",
                "attempts": 1, "seconds": 0.3,
                "backend_seconds": 0.25 if mode == "ok" else None,
                "error": None if mode == "ok" else "wrong_gradient_shape",
                "request_sha256": "x" * 64,
                "energy": VALID_RECORD["energy"] if mode == "ok" else None,
                "g_abs_max": 0.3 if mode == "ok" else None,
                "receipt_dir": str(rd), "attempt_log": [{"attempt": 1}]}

    return compute


def latest_ledger_entry(out_root: Path) -> dict:
    ledger = json.loads((out_root / "compute_ledger.json").read_text())
    return ledger["runs"][-1]


# ---------------------------------------------------------------------------
# Receipt validation
# ---------------------------------------------------------------------------


def test_validation_rejects_wrong_shape_nonfinite_and_missing():
    import copy
    request_body = {"geometry": [[[0.0, 0.0, 0.0]] * 3]}
    binding = digest(request_body)
    request = dict(request_body, request_sha256=binding)
    good = dict(success=True, energy=-1.0,
                gradient_hartree_per_angstrom=[[0.1, 0.0, 0.0]] * 3,
                request_sha256=binding)
    assert validate_receipt(request, good, 3) == (True, None)

    wrong_shape = copy.deepcopy(good)
    wrong_shape["gradient_hartree_per_angstrom"] = [[0.1, 0.0]] * 3
    ok, reason = validate_receipt(request, wrong_shape, 3)
    assert not ok and "wrong_shape" in reason or "not_3_components" in reason

    wrong_atoms = copy.deepcopy(good)
    wrong_atoms["gradient_hartree_per_angstrom"] = [[0.1, 0.0, 0.0]] * 4
    ok, reason = validate_receipt(request, wrong_atoms, 3)
    assert not ok and "wrong_shape" in reason

    nonfinite_e = copy.deepcopy(good)
    nonfinite_e["energy"] = float("nan")
    ok, reason = validate_receipt(request, nonfinite_e, 3)
    assert not ok and "energy" in reason

    nonfinite_g = copy.deepcopy(good)
    nonfinite_g["gradient_hartree_per_angstrom"][1][2] = float("inf")
    ok, reason = validate_receipt(request, nonfinite_g, 3)
    assert not ok and "not_finite" in reason

    missing_g = copy.deepcopy(good)
    missing_g["gradient_hartree_per_angstrom"] = None
    ok, reason = validate_receipt(request, missing_g, 3)
    assert not ok and "missing_or_wrong_shape" in reason

    missing_sha = copy.deepcopy(good)
    missing_sha["request_sha256"] = None
    ok, reason = validate_receipt(request, missing_sha, 3)
    assert not ok and "request_sha256_mismatch" in reason

    failed = copy.deepcopy(good)
    failed["success"] = False
    ok, reason = validate_receipt(request, failed, 3)
    assert not ok and "success_not_true" in reason

    bad_binding = copy.deepcopy(request)
    bad_binding["geometry"] = [[[9.9] * 3] * 3]
    ok, reason = validate_receipt(bad_binding, good, 3)
    assert not ok and "binding_mismatch" in reason


def test_no_zero_fill_invalid_receipt_registered_as_failure(
        coverage_dir: Path, tmp_path: Path):
    out = tmp_path / "obs"
    # mode=bad writes a success=False receipt whose gradient is a zero
    # placeholder THE FAKE produced; the driver must register the failure and
    # never rewrite/zero-fill the receipt itself.
    calls: list[str] = []
    args = make_args(coverage_dir, out)
    entry = run_driver(args, compute_fn=fake_compute_factory("bad", calls))
    assert entry["failed"] == 3 and entry["computed"] == 0
    registry = (out / "failures.jsonl").read_text().splitlines()
    assert len(registry) == 3
    for line in registry:
        record = json.loads(line)
        assert record["digest"] and record["attempts"] == 1
        assert record["error"]
    # receipts on disk keep exactly what the fake wrote (no driver rewrite)
    payloads = {p["digest"]: p for p in build_node_payloads(
        coverage_dir / "coverage_nodes.json")
        if p["cache_status"] != "cache_hit"}
    for digest_hex, payload in payloads.items():
        rd = receipt_dir(out, digest_hex)
        stored = json.loads((rd / "result.json").read_text())
        assert stored["success"] is False
        assert len(stored["gradient_hartree_per_angstrom"]) == payload["n_atoms"]
        assert "pes2ts_units" not in stored  # never annotated as valid


# ---------------------------------------------------------------------------
# Node selection + resume
# ---------------------------------------------------------------------------


def test_selection_skips_cache_hit_and_existing_valid_receipt(
        coverage_dir: Path, tmp_path: Path):
    out = tmp_path / "obs"
    payloads = build_node_payloads(coverage_dir / "coverage_nodes.json")
    by_digest = {p["digest"]: p for p in payloads}
    cache_hits = [p for p in payloads if p["cache_status"] == "cache_hit"]
    to_compute = [p for p in payloads if p["cache_status"] == "to_compute"]
    assert len(cache_hits) == 1 and len(to_compute) == 3
    # a pre-existing VALID receipt (shared TS node) is resume-skipped
    write_receipt(out, to_compute[0],
                  fake_backend_record(None, energy=VALID_RECORD["energy"],
                                      gradient=VALID_RECORD["gradient"]))
    ok, reason = receipt_state(out, to_compute[0])
    assert ok and reason is None

    calls: list[str] = []
    entry = run_driver(make_args(coverage_dir, out),
                       compute_fn=fake_compute_factory("ok", calls))
    assert sorted(calls) == sorted(p["digest"] for p in to_compute[1:])
    assert entry["skipped_cache"] == 1
    assert entry["skipped_resume"] == 1
    assert entry["attempted"] == 2 and entry["computed"] == 2 and entry["failed"] == 0


def test_resume_failed_receipt_skipped_then_retried_only_with_flag(
        coverage_dir: Path, tmp_path: Path):
    out = tmp_path / "obs"
    # run 1: every to_compute node fails (backend-shaped failed receipts)
    run_driver(make_args(coverage_dir, out),
               compute_fn=fake_compute_factory("bad"))
    registry_lines = (out / "failures.jsonl").read_text().splitlines()
    assert len(registry_lines) == 3
    failed_digests = {json.loads(l)["digest"] for l in registry_lines}

    # run 2 (resume): registered failures are skipped, never silently retried
    calls: list[str] = []
    entry = run_driver(make_args(coverage_dir, out),
                       compute_fn=fake_compute_factory("ok", calls))
    assert calls == []
    assert entry["attempted"] == 0
    assert entry["skipped_registered_failure"] == 3
    assert entry["computed"] == 0
    assert len((out / "failures.jsonl").read_text().splitlines()) == 3

    # run 3 (--retry-failed): the failed receipts are moved aside and retried
    args = make_args(coverage_dir, out, retry_failed=True)
    entry = run_driver(args, compute_fn=fake_compute_factory("ok", calls))
    assert sorted(calls) == sorted(failed_digests)
    assert entry["attempted"] == 3 and entry["computed"] == 3
    assert entry["skipped_resume_failed"] == 3
    assert entry["failed"] == 0
    # evidence of the failed attempts preserved, not deleted
    quarantine = list((out / "failed_attempts").rglob("result.json"))
    assert len(quarantine) == 3
    for path in quarantine:
        assert json.loads(path.read_text())["success"] is False
    # a further resume run finds everything valid
    entry = run_driver(make_args(coverage_dir, out),
                       compute_fn=fake_compute_factory("ok"))
    assert entry["attempted"] == 0 and entry["skipped_resume"] == 3


def test_binding_mismatch_refuses_before_any_compute(coverage_dir: Path,
                                                     tmp_path: Path):
    coverage_path = coverage_dir / "coverage_nodes.json"
    document = json.loads(coverage_path.read_text())
    document["nodes"][0]["computation_identity"] = "f" * 64
    coverage_path.write_text(json.dumps(document))
    with pytest.raises(CoverageBindingError, match="identity mismatch"):
        build_node_payloads(coverage_path)

    document = json.loads(coverage_path.read_text())
    document["nodes"][0]["computation_identity"] = (
        document["nodes"][0]["geometry_content_digest"] + "0" * 0)
    document["nodes"][0]["geometry_content_digest"] = "e" * 64
    coverage_path.write_text(json.dumps(document))
    with pytest.raises(CoverageBindingError, match="geometry digest mismatch"):
        build_node_payloads(coverage_path)


def test_shared_ts_state_refs_must_digest_identically(coverage_dir: Path,
                                                      tmp_path: Path):
    coverage_path = coverage_dir / "coverage_nodes.json"
    document = json.loads(coverage_path.read_text())
    ts_node = next(n for n in document["nodes"] if n["is_ts"])
    assert len(ts_node["state_refs"]) == 2
    # corrupt one branch's TS state coords -> payload build must refuse
    teacher_path = coverage_dir / "RXN_T.teacher.json"
    teacher = json.loads(teacher_path.read_text())
    for state in teacher["states"]:
        if state["state_id"] == "RXN_T|B|0001":
            state["coordinates"][0][0] += 1e-6
    teacher_path.write_text(json.dumps(teacher))
    with pytest.raises(CoverageBindingError, match="dedupe invariant"):
        build_node_payloads(coverage_path)


# ---------------------------------------------------------------------------
# Ledger + progress + limit
# ---------------------------------------------------------------------------


def test_ledger_accounting_across_two_runs(coverage_dir: Path,
                                           tmp_path: Path):
    out = tmp_path / "obs"
    args = make_args(coverage_dir, out, limit=2)
    first = run_driver(args, compute_fn=fake_compute_factory("ok"))
    assert first["attempted"] == 2 and first["computed"] == 2
    assert len(first["per_receipt_seconds"]) == 2
    assert all(v > 0 for v in first["per_receipt_seconds"].values())

    second = run_driver(make_args(coverage_dir, out),
                        compute_fn=fake_compute_factory("ok"))
    assert second["attempted"] == 1 and second["computed"] == 1
    assert second["skipped_resume"] == 2

    ledger = json.loads((out / "compute_ledger.json").read_text())
    assert ledger["schema_version"].startswith("pes2ts_dual_ts_obs_compute")
    assert [r["run_id"] for r in ledger["runs"]] == [first["run_id"],
                                                     second["run_id"]]
    assert second["cumulative_wall_seconds"] >= (
        first["wall_seconds"] + second["wall_seconds"] - 1e-6)
    assert ledger["runs"][0]["limit"] == 2 and ledger["runs"][1]["limit"] is None
    assert ledger["runs"][0]["workers"] == 1


def test_progress_jsonl_records_inspectable_events(coverage_dir: Path,
                                                   tmp_path: Path):
    out = tmp_path / "obs"
    run_driver(make_args(coverage_dir, out),
               compute_fn=fake_compute_factory("ok"))
    events = [json.loads(line)
              for line in (out / "progress.jsonl").read_text().splitlines()]
    statuses = [e["status"] for e in events]
    assert statuses.count("skipped_cache") == 1
    assert statuses.count("computed") == 3
    assert all(e["ts"] and e["run_id"] for e in events)
    for event in events:
        if event["status"] == "computed":
            assert event["node_digest"] and event["seconds"] >= 0


def test_limit_bounds_new_attempts_not_resume(coverage_dir: Path,
                                              tmp_path: Path):
    out = tmp_path / "obs"
    calls: list[str] = []
    run_driver(make_args(coverage_dir, out, limit=1),
               compute_fn=fake_compute_factory("ok", calls))
    assert len(calls) == 1
    calls.clear()
    entry = run_driver(make_args(coverage_dir, out, limit=1),
                       compute_fn=fake_compute_factory("ok", calls))
    assert len(calls) == 1
    assert entry["skipped_resume"] == 1 and entry["skipped_cache"] == 1
    calls.clear()
    entry = run_driver(make_args(coverage_dir, out),
                       compute_fn=fake_compute_factory("ok", calls))
    assert len(calls) == 1 and entry["attempted"] == 1


# ---------------------------------------------------------------------------
# Unit-conversion recording on a canned result
# ---------------------------------------------------------------------------


def test_units_annotation_records_native_units_and_conversion(
        coverage_dir: Path, tmp_path: Path):
    out = tmp_path / "obs"
    payloads = build_node_payloads(coverage_dir / "coverage_nodes.json")
    payload = next(p for p in payloads if p["cache_status"] == "to_compute")
    write_receipt(out, payload,
                  fake_backend_record(None, energy=VALID_RECORD["energy"],
                                      gradient=VALID_RECORD["gradient"]))
    rd = receipt_dir(out, payload["digest"])
    before = json.loads((rd / "result.json").read_text())

    annotate_receipt(rd, payload, {"root": "/fake/acp",
                                   "python": "/fake/python",
                                   "config_path": "/fake/acp.yaml",
                                   "method": "GFN2-xTB",
                                   "timeout_seconds": 120.0, "nproc": 2})
    result = json.loads((rd / "result.json").read_text())
    units = result["pes2ts_units"]
    assert units["energy"] == "hartree"
    assert units["gradient_hartree_per_angstrom"] == "hartree/angstrom"
    assert units["gradient_native_unit"] == "hartree/bohr"
    assert units["BOHR_ANGSTROM"] == units_annotation()["BOHR_ANGSTROM"]
    assert "BOHR_ANGSTROM" in units["conversion"]
    # additive only: every pre-existing key and value is untouched
    for key, value in before.items():
        assert result[key] == value
    # the annotation must survive the resume validator (pes2ts_ keys excluded
    # from the binding re-digest)
    request = json.loads((rd / "request.json").read_text())
    assert "pes2ts_computation_identity" in request
    ok, reason = validate_receipt(request, result, payload["n_atoms"])
    assert ok and reason is None
    identity = request["pes2ts_computation_identity"]
    assert identity["computation_identity"] == payload["digest"]
    assert identity["state_refs"] == payload["state_refs"]


def test_per_qc_seconds_recorded_from_backend_duration(coverage_dir: Path,
                                                       tmp_path: Path):
    out = tmp_path / "obs"
    entry = run_driver(make_args(coverage_dir, out),
                       compute_fn=fake_compute_factory("ok"))
    assert set(entry["per_receipt_seconds"]) == {
        p["digest"] for p in build_node_payloads(
            coverage_dir / "coverage_nodes.json")
        if p["cache_status"] == "to_compute"}
    assert all(0.0 < v < 60.0 for v in entry["per_receipt_seconds"].values())
    assert set(entry["per_receipt_wall_seconds"]) == set(
        entry["per_receipt_seconds"])
