"""Synthetic tests for WP2-C dual-TS observation assembly.

No real data reads, no ACP/xTB, no network: a one-reaction fit50-like table
(4 coverage nodes: 3 to_compute + 1 v1 cache hit over a shared TS) is built
in a tmp dir, with receipts written through the REAL frozen request builder
(``build_gradient_request`` + ``digest``) so binding exercises the same rules
the real receipts satisfy.  Covered: receipt→ref binding, missing-receipt
null+reason + gaps registration, zero-fill rejection, unit conversion
verify/apply/mismatch, typed full-coverage gate, stale-geometry binding
gate, manifest round-trip.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pes2ts_core.generation.planning.gradient_evidence import (  # noqa: E402
    BOHR_ANGSTROM,
)
from pes2ts_core.generation.planning.synchronized_path import digest  # noqa: E402
from pes2ts_core.integration.acp.gradient_backend import write_json  # noqa: E402
from pes2ts_core.integration.acp.orca_gradient_request import (  # noqa: E402
    build_gradient_request,
)
from pes2ts_core.pesn.contracts.dual_ts import (  # noqa: E402
    DualEndpointTask,
    DualTSTeacher,
    EndpointBond,
    EndpointGraph,
    EpisodeRecord,
    BranchRecord,
    ObservationRef,
    TeacherLabel,
    TeacherState,
)
from pes2ts_core.pesn.offline.data.dual_ts_assemble import (  # noqa: E402
    AssembleError,
    CoverageIncompleteError,
    bind_node_observation,
    locate_receipt,
    sha256_file,
    validate_receipt_pair,
)
from pes2ts_core.pesn.offline.data.dual_ts_fit50 import (  # noqa: E402
    atom_order_hash,
    computation_identity,
    geometry_content_digest,
)

XTB_BLOCKS = ['%xtb\n XTBINPUTSTRING "--iterations 2000"\nend']
Z = [1, 8, 1]
MAPS = [f"m{i + 1:04d}" for i in range(len(Z))]
ELEMENTS = ["H", "O", "H"]
METHOD = "GFN2-xTB"

NATIVE_G = [[0.0101, -0.0202, 0.0303], [0.0, 0.0051, -0.0051],
            [-0.0101, 0.0152, -0.0202]]  # hartree/bohr


def _coords(seed: int) -> list[list[float]]:
    rng = np.random.default_rng(seed)
    return [[round(float(v), 10) for v in row]
            for row in rng.normal(scale=0.5, size=(3, 3))]


def _teacher_payload(coords: dict[str, list[list[float]]]) -> dict:
    """Contract-valid one-reaction teacher: A0→A1→TS, B0→TS (shared TS)."""
    def state(sid, branch, episode, step, terminal, nxt):
        return TeacherState(
            state_id=f"RXN_T|{branch}|{step:04d}",
            episode_id=episode, branch_id=f"RXN_T#br{branch}",
            step_index=step,
            atomic_numbers=np.asarray(Z, dtype=int),
            coordinates=np.asarray(coords[sid], dtype=float),
            atom_map_ids=tuple(MAPS), method_id=METHOD,
            observation=None,
            observation_missing_reason=(
                "teacher_frame_isolated" if terminal
                else "observation_not_exported"),
            boundary_type="teacher_endpoint" if terminal else None,
            next_state_id=nxt,
        )

    ts = np.asarray(coords["TS"], dtype=float)
    states = [
        state("A0", "A", "RXN_T#epA0", 0, False, "RXN_T|A|0001"),
        state("A1", "A", "RXN_T#epA0", 1, False, "RXN_T|A|0002"),
        state("TS", "A", "RXN_T#epA0", 2, True, None),
        state("B0", "B", "RXN_T#epB0", 0, False, "RXN_T|B|0001"),
        state("TS", "B", "RXN_T#epB0", 1, True, None),
    ]

    def label(s: TeacherState) -> TeacherLabel:
        terminal = s.boundary_type == "teacher_endpoint"
        if terminal:
            d = np.zeros((3, 3))
            a = None
            arc = 0.0
        else:
            d = ts - s.coordinates
            nxt = next(x for x in states if x.state_id == s.next_state_id)
            a = np.asarray(nxt.coordinates, dtype=float) - s.coordinates
            arc = 5.0 - 2.5 * s.step_index
        return TeacherLabel(state_id=s.state_id, ts_displacement=d,
                            remaining_arc_length=arc,
                            teacher_endpoint=terminal, action_delta=a)

    teacher = DualTSTeacher(
        reaction_id="RXN_T", reaction_group_id="RXN_T",
        branches=(
            BranchRecord(branch_id="RXN_T#brA", endpoint_label="A",
                         reaction_id="RXN_T", episode_ids=("RXN_T#epA0",)),
            BranchRecord(branch_id="RXN_T#brB", endpoint_label="B",
                         reaction_id="RXN_T", episode_ids=("RXN_T#epB0",)),
        ),
        episodes=(
            EpisodeRecord(episode_id="RXN_T#epA0", branch_id="RXN_T#brA",
                          reaction_id="RXN_T",
                          start_state_id="RXN_T|A|0000",
                          budget_version="bv1", environment_version="ev1",
                          reward_version="rv1"),
            EpisodeRecord(episode_id="RXN_T#epB0", branch_id="RXN_T#brB",
                          reaction_id="RXN_T",
                          start_state_id="RXN_T|B|0000",
                          budget_version="bv1", environment_version="ev1",
                          reward_version="rv1"),
        ),
        states=tuple(states),
        labels=tuple(label(s) for s in states),
        arc_length_metric_version="mass_weighted_arc_v1_diag_masses",
    )
    return teacher.to_payload()


def _task_payload() -> dict:
    def graph() -> EndpointGraph:
        return EndpointGraph(
            atomic_numbers=np.asarray(Z, dtype=int),
            atom_map_ids=tuple(MAPS),
            bonds=(EndpointBond(i=0, j=1, bond_order=1.0),
                   EndpointBond(i=1, j=2, bond_order=2.0)),
        )

    task = DualEndpointTask(
        reaction_id="RXN_T", reaction_group_id="RXN_T",
        endpoints=(graph(), graph()), method_id=METHOD,
        source="synthetic",
    )
    return task.canonical_payload()


def _node(coords, refs, cache_status="to_compute", receipt_path=None,
          is_ts=False) -> dict:
    geom = geometry_content_digest(np.asarray(coords, dtype=float))
    order = atom_order_hash(np.asarray(Z, dtype=int), MAPS)
    identity = computation_identity(geom, order, METHOD, 0, 1,
                                    "eg_single_point")
    return {
        "node_id": f"RXN_T|node|{identity['digest'][:12]}",
        "reaction_id": "RXN_T",
        "reaction_group_id": "RXN_T",
        "geometry_content_digest": geom,
        "atom_order_hash": order,
        "method": METHOD,
        "charge": 0,
        "multiplicity": 1,
        "operation": "eg_single_point",
        "computation_identity": identity["digest"],
        "state_refs": refs,
        "is_ts": is_ts,
        "inserted": False,
        "inserted_state_ids": [],
        "cache": {"status": cache_status, "receipt_path": receipt_path,
                  "request_sha256": None, "duration_seconds": None,
                  "n_receipts": 1 if cache_status == "cache_hit" else 0},
    }


def _request(coords, *, nproc=1, annotate=True, energy=-19.0995,
             native=None, converted=..., units=True, settings=True):
    request = build_gradient_request(
        geometry=coords, elements=ELEMENTS, method=METHOD, basis="",
        charge=0, multiplicity=1, route_extras=["EnGrad"],
        timeout_seconds=120, nproc=nproc, extra_blocks=XTB_BLOCKS,
        scf_convergence="tight", output_name="grad")
    binding = digest(request)
    request = {**request, "request_sha256": binding,
               "acp_request_sha256": "acp-" + binding[:16]}
    if annotate:
        identity = computation_identity(
            geometry_content_digest(np.asarray(coords, dtype=float)),
            atom_order_hash(np.asarray(Z, dtype=int), MAPS),
            METHOD, 0, 1, "eg_single_point")
        request["pes2ts_computation_identity"] = {
            "scheme": "pes2ts_dual_ts_wp2_computation_identity_v0",
            "computation_identity": identity["digest"],
            "driver_version": "pes2ts_dual_ts_obs_compute_v0",
        }
        if settings:
            request["pes2ts_settings"] = {
                "scheme": "pes2ts_dual_ts_obs_settings_v0",
                "settings_digest": digest({"workflow": "OrcaGradient",
                                           "method": METHOD, "nproc": nproc}),
            }
    native = NATIVE_G if native is None else native
    result = {
        "success": True,
        "failure_class": None,
        "request_sha256": binding,
        "energy": energy,
        "gradient_evidence": {"status": "bound",
                              "gradient_hartree_per_bohr": native,
                              "energy_hartree": energy},
        "gradient_hartree_per_angstrom": (
            (np.asarray(native) / BOHR_ANGSTROM).tolist()
            if converted is ... else converted),
        "duration_seconds": 0.42,
        "acp": {"workflow": "OrcaGradient", "status": "completed"},
        "method": METHOD,
        "normal_termination": True,
    }
    if annotate and units:
        result["pes2ts_units"] = {
            "scheme": "pes2ts_dual_ts_obs_units_v0",
            "BOHR_ANGSTROM": BOHR_ANGSTROM,
        }
    return request, result


def _write_receipt(out_root: Path, digest_hex: str, request: dict,
                   result: dict) -> Path:
    rd = out_root / digest_hex[:2] / digest_hex
    rd.mkdir(parents=True, exist_ok=True)
    write_json(rd / "request.json", request)
    write_json(rd / "result.json", result)
    return rd


@pytest.fixture()
def env(tmp_path: Path) -> dict:
    """fit50-like table + receipts.  A1 has NO receipt by default (used by
    the partial/coverage-gate tests); ``complete=True`` adds it."""
    coords = {sid: _coords(i) for i, sid
              in enumerate(["A0", "A1", "TS", "B0"])}
    fit50 = tmp_path / "fit50"
    fit50.mkdir()
    (fit50 / "RXN_T.teacher.json").write_text(
        json.dumps(_teacher_payload(coords), ensure_ascii=False, indent=1),
        encoding="utf-8")
    (fit50 / "RXN_T.task.json").write_text(
        json.dumps(_task_payload(), ensure_ascii=False, indent=1),
        encoding="utf-8")

    def ref(sid, branch, step, terminal=False):
        return {"state_id": f"RXN_T|{branch}|{step:04d}",
                "branch_id": f"RXN_T#br{branch}", "step_index": step,
                "is_terminal": terminal, "inserted": False}

    v1_dir = tmp_path / "v1_cache" / "RXN_T" / "direct-RXN_T" / \
        "eval-08a37001bef2ebab"
    v1_dir.mkdir(parents=True)
    ts_request, ts_result = _request(
        coords["TS"], nproc=2, annotate=False)  # v1: no pes2ts_ annotations
    write_json(v1_dir / "request.json", ts_request)
    write_json(v1_dir / "result.json", ts_result)

    nodes = [
        _node(coords["A0"], [ref("A0", "A", 0)]),
        _node(coords["A1"], [ref("A1", "A", 1)]),
        _node(coords["TS"],
              [ref("TS", "A", 2, terminal=True),
               ref("TS", "B", 1, terminal=True)],
              cache_status="cache_hit", receipt_path=str(v1_dir), is_ts=True),
        _node(coords["B0"], [ref("B0", "B", 0)]),
    ]
    (fit50 / "coverage_nodes.json").write_text(
        json.dumps({"schema_version":
                    "pes2ts_dual_ts_wp2_coverage_nodes_v0",
                    "identity_scheme": {"canonical_json": "stable_json_dumps"},
                    "nodes": nodes}, ensure_ascii=False, indent=1),
        encoding="utf-8")

    lineage = tmp_path / "splits" / "lineage.json"
    lineage.parent.mkdir()
    lineage.write_text('{"ok": true}\n', encoding="utf-8")
    (fit50 / "coverage_summary.json").write_text(json.dumps({
        "schema_version": "pes2ts_dual_ts_wp2_coverage_summary_v0",
        "tool_version": "pes2ts_dual_ts_fit50_wp2_v0",
        "generated_utc": "2026-10-16T00:00:00+00:00",
        "lineage_artifact": {"path": str(lineage),
                             "sha256": sha256_file(lineage)},
        "reactions": {"wave_fit_total": 1, "attempted": 1, "ok": 1,
                      "excluded": 0, "exclusion_codes": {}},
        "branches": {"count": 2, "states_total": 5,
                     "states_per_reaction_avg": 5.0},
        "insertions": {"reactions_with_insertion": 1,
                       "nodes_inserted_total": 2},
    }, indent=1), encoding="utf-8")
    (fit50 / "gaps_registry.json").write_text(json.dumps({
        "schema_version": "pes2ts_dual_ts_gaps_registry_v1",
        "reactions": [{"reaction_id": "RXN_T", "reaction_group_id": "RXN_T",
                       "partition": "fit", "status": "ok", "reason": "",
                       "items": {
                           "bond_cross_check": {
                               "outcome": "disputed",
                               "disputed_bonds": [{"side": "R", "i": 5,
                                                   "j": 7}]},
                           "mapping_problems": [
                               "bond_authority_disputed: changed bond (5,7)"],
                       }}]}, indent=1), encoding="utf-8")

    obs = tmp_path / "observations"
    obs.mkdir()
    ts_digest = nodes[2]["computation_identity"]
    for name, node in (("A0", nodes[0]), ("B0", nodes[3])):
        request, result = _request(coords[name], nproc=1)
        _write_receipt(obs, node["computation_identity"], request, result)
    write_json(obs / "compute_ledger.json", {
        "schema_version": "pes2ts_dual_ts_obs_compute_ledger_v0",
        "runs": [{"run_id": "run-1", "computed": 2, "failed": 0,
                  "skipped_resume": 0, "skipped_cache": 1,
                  "interrupted": False, "wall_seconds": 1.25,
                  "settings_digest": "d" * 64}],
    })
    return {
        "fit50": fit50, "obs": obs, "nodes": nodes, "coords": coords,
        "ts_digest": ts_digest, "v1_dir": v1_dir, "tmp": tmp_path,
    }


def _add_receipt(env: dict, name: str, **request_kwargs) -> str:
    node = env["nodes"][1] if name == "A1" else None
    assert node is not None and name == "A1"
    request, result = _request(env["coords"]["A1"], **request_kwargs)
    rd = _write_receipt(env["obs"], node["computation_identity"],
                        request, result)
    return node["computation_identity"]


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_receipt_binds_into_teacher_refs(env):
    from pes2ts_core.pesn.offline.data.dual_ts_assemble import assemble

    _add_receipt(env, "A1")
    out = env["tmp"] / "out"
    manifest = assemble(env["fit50"], env["obs"], out,
                        require_full_coverage=True)
    teacher = json.loads(
        (out / "RXN_T.teacher.json").read_text(encoding="utf-8"))
    bound = [s for s in teacher["states"] if s["observation"] is not None]
    assert len(bound) == 5                      # all states bound
    for state in bound:
        obs = state["observation"]
        assert obs["status"] == "ok"
        assert obs["missing_reason"] is None
        assert state["observation_missing_reason"] is None
        assert obs["method_id"] == METHOD
        assert obs["software"] == "OrcaGradient"
        assert len(obs["gradient"]) == 3 and len(obs["gradient"][0]) == 3
        assert state["observation_computation_identity"]
        assert Path(state["observation_receipt_path"]).is_dir()
    # shared TS: both terminal states reference the SAME cache receipt
    ts_states = [s for s in bound if s["boundary_type"] == "teacher_endpoint"]
    assert len(ts_states) == 2
    assert (ts_states[0]["observation_computation_identity"]
            == ts_states[1]["observation_computation_identity"]
            == env["ts_digest"])
    assert (ts_states[0]["observation_receipt_path"]
            == str(env["v1_dir"]))
    # task payload is a byte copy; coverage artifacts copied
    assert (out / "RXN_T.task.json").read_bytes() == \
        (env["fit50"] / "RXN_T.task.json").read_bytes()
    assert (out / "coverage_nodes.json").read_bytes() == \
        (env["fit50"] / "coverage_nodes.json").read_bytes()
    # index summary: 4/4 nodes, attribution + provenance
    index = json.loads(
        (out / "observation_index.json").read_text(encoding="utf-8"))
    summary = index["summary"]
    assert summary["total_nodes"] == 4 and summary["bound_nodes"] == 4
    assert summary["missing_nodes"] == 0
    assert summary["attribution_counts"] == {"wp2_computed": 3,
                                             "v1_cache": 1}
    assert summary["per_reaction_complete"] == "1/1"
    assert summary["settings_digest_counts"]["unrecorded_v1_cache"] == 1
    assert summary["nproc_provenance"] == {"1": 3, "2": 1}
    # v1 receipt has no settings annotation — honest null, software recorded
    ts_record = next(r for r in index["nodes"]
                     if r["node_digest"] == env["ts_digest"])
    assert ts_record["settings_digest"] is None
    assert ts_record["settings_provenance"] == "receipt_settings_unrecorded"
    assert ts_record["conversion"]["source"] == \
        "canonical_gradient_evidence_constant"
    # annotated new receipt carries settings + unit source
    a0 = next(r for r in index["nodes"]
              if r["node_id"].endswith(
                  env["nodes"][0]["computation_identity"][:12]))
    assert a0["settings_digest"]
    assert a0["settings_provenance"] == "receipt_pes2ts_settings"
    assert a0["conversion"]["source"] == "receipt_pes2ts_units"
    assert a0["driver_version"] == "pes2ts_dual_ts_obs_compute_v0"
    # manifest coverage audit agrees with the index
    assert manifest["coverage_audit"]["coverage"] == "4/4"
    assert manifest["coverage_audit"]["per_reaction_complete"] == "1/1"
    assert manifest["denominators"]["states_bound_to_observations"] == 5


def test_binding_gate_rejects_stale_geometry(env):
    node = dict(env["nodes"][0])
    node["geometry_content_digest"] = "0" * 64
    states = {s["state_id"]: s for s in json.loads(
        (env["fit50"] / "RXN_T.teacher.json").read_text(encoding="utf-8")
    )["states"]}
    with pytest.raises(AssembleError, match="BINDING_MISMATCH"):
        bind_node_observation(node, states, env["obs"])


def test_missing_receipt_keeps_null_reason_and_registers_gaps(env):
    from pes2ts_core.pesn.offline.data.dual_ts_assemble import (
        assemble_observations,
        write_observed_export,
    )
    assembled = assemble_observations(env["fit50"], env["obs"],
                                      require_full_coverage=False)
    out = env["tmp"] / "out_partial"
    write_observed_export(env["fit50"], env["obs"], out, assembled)
    teacher = json.loads(
        (out / "RXN_T.teacher.json").read_text(encoding="utf-8"))
    a1 = next(s for s in teacher["states"]
              if s["state_id"] == "RXN_T|A|0001")
    # null + ORIGINAL registered reason preserved — never zero-filled
    assert a1["observation"] is None
    assert a1["observation_missing_reason"] == "observation_not_exported"
    assert "observation_computation_identity" not in a1
    bound = [s for s in teacher["states"] if s["observation"] is not None]
    assert len(bound) == 4
    gaps = json.loads(
        (out / "gaps_registry_observed.json").read_text(encoding="utf-8"))
    addendum = gaps["wp2_assemble"]["observation_binding"]
    assert addendum["states_still_missing"] == 1
    assert addendum["zero_fill_used"] is False
    assert addendum["missing"][0]["state_id"] == "RXN_T|A|0001"
    assert addendum["missing"][0]["receipt_status"] == "missing"
    index = json.loads(
        (out / "observation_index.json").read_text(encoding="utf-8"))
    assert index["summary"]["missing_nodes"] == 1
    assert index["missing_nodes"][0]["node_digest"] == \
        env["nodes"][1]["computation_identity"]
    assert index["per_reaction"]["RXN_T"]["complete"] is False
    assert index["per_reaction"]["RXN_T"]["bound"] == 3


def test_zero_filled_receipt_never_binds(env):
    from pes2ts_core.pesn.contracts.dual_ts import DualTSContractError
    from pes2ts_core.pesn.offline.data.dual_ts_assemble import assemble

    _add_receipt(env, "A1", energy=0.0,
                 native=[[0.0, 0.0, 0.0]] * 3, converted=[[0.0] * 3] * 3)
    # the contract-level zero-fill marker fires on the rewritten payload —
    # the assembly aborts loudly instead of exporting zeros
    with pytest.raises(DualTSContractError, match="OBS_ZERO_FILLED"):
        assemble(env["fit50"], env["obs"], env["tmp"] / "out_zero",
                 require_full_coverage=True)
    # and directly: the zero observation can never validate
    with pytest.raises(DualTSContractError, match="OBS_ZERO_FILLED"):
        ObservationRef(
            status="ok", energy_hartree=0.0,
            gradient_hartree_per_angstrom=np.zeros((3, 3)),
            method_id=METHOD, software="OrcaGradient",
        ).validate()


def test_unit_conversion_verify_apply_and_mismatch(env):
    states = {s["state_id"]: s for s in json.loads(
        (env["fit50"] / "RXN_T.teacher.json").read_text(encoding="utf-8")
    )["states"]}
    node_a0 = env["nodes"][0]
    record = bind_node_observation(node_a0, states, env["obs"])
    assert record["receipt_status"] == "ok"
    assert record["conversion"]["verified"] is True
    assert record["conversion"]["applied"] is False
    native = np.asarray(NATIVE_G)
    g = np.asarray(record["_gradient_rows"])
    assert np.allclose(g, native / BOHR_ANGSTROM, rtol=1e-12)
    # apply path: receipt with native only -> conversion applied + validated
    request, result = _request(env["coords"]["A1"], converted=None)
    _write_receipt(env["obs"], env["nodes"][1]["computation_identity"],
                   request, result)
    record = bind_node_observation(env["nodes"][1], states, env["obs"])
    assert record["receipt_status"] == "ok"
    assert record["conversion"]["applied"] is True
    assert np.allclose(np.asarray(record["_gradient_rows"]),
                       native / BOHR_ANGSTROM, rtol=1e-12)
    # mismatch path: tampered converted array -> typed invalid receipt
    bad = (native / BOHR_ANGSTROM * 1.1).tolist()
    request, result = _request(env["coords"]["B0"], converted=bad)
    _write_receipt(env["obs"], env["nodes"][3]["computation_identity"],
                   request, result)
    record = bind_node_observation(env["nodes"][3], states, env["obs"])
    assert record["receipt_status"].startswith("invalid:conversion_mismatch")
    assert record["energy_hartree"] is None and record["_gradient_rows"] is None


def test_full_coverage_gate_typed_error_and_no_export(env):
    # A1 intentionally has no receipt in this fixture
    with pytest.raises(CoverageIncompleteError) as excinfo:
        from pes2ts_core.pesn.offline.data.dual_ts_assemble import (
            assemble_observations,
        )
        assemble_observations(env["fit50"], env["obs"],
                              require_full_coverage=True)
    assert excinfo.value.code == "COVERAGE_INCOMPLETE"
    assert len(excinfo.value.missing) == 1
    assert excinfo.value.missing[0]["node_digest"] == \
        env["nodes"][1]["computation_identity"]
    assert not (env["tmp"] / "out_never").exists()


def test_locate_receipt_and_driver_parity(env):
    # content-addressed path for to_compute, recorded path for cache hits
    assert locate_receipt(env["nodes"][0], env["obs"]) == \
        env["obs"] / env["nodes"][0]["computation_identity"][:2] / \
        env["nodes"][0]["computation_identity"]
    assert locate_receipt(env["nodes"][2], env["obs"]) == env["v1_dir"]
    # validate_receipt_pair (driver-compatible) accepts a real written pair
    request = json.loads(
        (env["v1_dir"] / "request.json").read_text(encoding="utf-8"))
    result = json.loads(
        (env["v1_dir"] / "result.json").read_text(encoding="utf-8"))
    ok, reason = validate_receipt_pair(request, result, n_atoms=3)
    assert ok and reason is None
    # tampered result sha breaks the binding
    bad = dict(result, request_sha256="f" * 64)
    ok, reason = validate_receipt_pair(request, bad, n_atoms=3)
    assert not ok and reason == "result_request_sha256_mismatch"


def test_manifest_round_trip(env):
    from pes2ts_core.pesn.offline.data.dual_ts_assemble import assemble

    _add_receipt(env, "A1")
    out = env["tmp"] / "out"
    manifest = assemble(env["fit50"], env["obs"], out)
    rt = json.loads(
        (out / "wp2_manifest.json").read_text(encoding="utf-8"))
    # bound digests recompute from the files on disk
    index_path = out / "observation_index.json"
    assert rt["observation_index"]["sha256"] == sha256_file(index_path)
    assert rt["observation_index"]["summary"]["bound_nodes"] == 4
    assert rt["gaps_registry"]["sha256"] == \
        sha256_file(out / "gaps_registry.json")
    assert rt["gaps_registry"]["observed_addendum"][
        "states_still_missing"] == 0
    per = rt["teacher_table"]["per_reaction_teacher_sha256"]
    assert per["RXN_T"] == sha256_file(out / "RXN_T.teacher.json")
    recombined = __import__("hashlib").sha256(
        "\n".join(f"{k}:{v}" for k, v in sorted(per.items()))
        .encode()).hexdigest()
    assert rt["teacher_table"]["teacher_table_sha256_combined"] == recombined
    # lineage artifact sha matches the recorded file
    assert rt["lineage_artifact"]["matches"] is True
    assert rt["lineage_artifact"]["sha256_recomputed"] == \
        rt["lineage_artifact"]["sha256_recorded"]
    # denominators + carry-forwards extracted from artifacts
    assert rt["denominators"] == {
        "reaction_groups": 1, "branches": 2, "states": 5,
        "states_from_coverage_summary": 5, "dual_sided_complete": "1/1",
        "states_bound_to_observations": 5,
        "states_missing_observations": 0,
    }
    carry = rt["carry_forwards"]
    assert carry["bond_cross_check"]["disputed"] == 1
    assert carry["disputed_reactions"][0]["reaction_id"] == "RXN_T"
    assert carry["disputed_reactions"][0]["disputed_bonds"] == [
        {"side": "R", "i": 5, "j": 7}]
    assert carry["insertions"]["nodes_inserted_total"] == 2
    # compute ledger totals bound (synthetic ledger file)
    totals = rt["compute_ledger"]["totals"]
    assert totals["runs"] == 1 and totals["computed"] == 2
    assert totals["failed"] == 0
    assert totals["skipped_cache_final_run"] == 1
    # tool versions carry the frozen constants
    tv = rt["tool_versions"]
    assert tv["bohr_angstrom"] == BOHR_ANGSTROM
    assert tv["assemble_tool"] == "pes2ts_dual_ts_assemble_wp2_v0"
    assert tv["coverage_tool"] == "pes2ts_dual_ts_fit50_wp2_v0"
