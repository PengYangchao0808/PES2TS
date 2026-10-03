"""Fake-ACP tests for the ACP-executed XTB_PATH backend (X2'-B).

Mirrors the fake-ACP-process pattern of ``test_acp_xtb_path_transport.py``:
a stub ``src/acp/cli.py`` stands in for the real ACP engine and writes a
minimal valid ``RESULT/`` tree (v2 manifest + ``pes_profile_v2`` + path
frames + a raw multi-frame ``xtbpath.xyz`` WITH ``energy:`` comments so the
frozen xTB parser accepts it).  Test hooks travel inside the request recipe
``extra_args`` (``--test-exit-code N``, ``--test-no-energy``,
``--test-bad-manifest``); a real ACP never sees these.  No real xTB binary
is ever invoked.

Covers: helper success (parsed frames + ACP provenance + ``g2_path_v1``
attempt keys present-but-None), helper transport failure and parse failure
(typed ``G2_XTB_FAILED``), backend ``run`` success/failure projections into
``TrajectoryRecord``, and ``prepare`` validation of the plan/materials
contract.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pytest

from pes2ts_core.g0.rejections import RejectionCode
from pes2ts_core.generation.execution.protocol import ExecutionBackend, METHOD_XTB_PATH
from pes2ts_core.generation.execution.record import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    TrajectoryRecord,
)
from pes2ts_core.generation.execution.xtb_path.acp_backend import (
    DIRECTION_FORWARD,
    XtbPathACPBackend,
    XtbPathAttemptOutcome,
    run_xtb_path_acp_attempt,
)
from pes2ts_core.generation.execution.xtb_path.xtb_output import Frame

START_XYZ = (
    "2\nPES2TS start\n"
    "H 0.000000 0.000000 0.000000\n"
    "H 0.740000 0.000000 0.000000\n"
)
END_XYZ = (
    "2\nPES2TS end\n"
    "H 0.000000 0.000000 0.000000\n"
    "H 1.500000 0.000000 0.000000\n"
)
FRAME_ENERGIES = (0.0, 2.5, 1.0)

#: Stub ACP CLI. Hooks arrive as recipe ``extra_args`` tokens the helper
#: forwards inside the frozen request (the fake CLI intercepts them).
_FAKE_CLI = '''\
import hashlib, json, pathlib, sys

args = sys.argv[1:]
exit_code = 0
bad_manifest = False
no_energy = False
request_path = pathlib.Path(args[args.index("--path-config") + 1])
output = pathlib.Path(args[args.index("--output") + 1])
request = json.loads(request_path.read_text(encoding="utf-8"))
assert request.get("schema_version") == "pes2ts_xtb_path_request_v1"
extra = request.get("recipe", {}).get("extra_args") or []
i = 0
while i < len(extra):
    if extra[i] == "--test-exit-code":
        exit_code = int(extra[i + 1]); i += 2; continue
    if extra[i] == "--test-bad-manifest":
        bad_manifest = True; i += 1; continue
    if extra[i] == "--test-no-energy":
        no_energy = True; i += 1; continue
    i += 1
if exit_code:
    raise SystemExit(exit_code)
result = output / "RESULT" / "pes_search"
frames_dir = result / "path_frames"
frames_dir.mkdir(parents=True, exist_ok=True)
profile = {"schema_version": "pes_profile_v2", "workflow": "XtbPathSearch",
           "status": "completed", "frames": [
               {"index": 0, "energy_rel_kcal": 0.0},
               {"index": 1, "energy_rel_kcal": 2.5},
               {"index": 2, "energy_rel_kcal": 1.0}]}
(result / "pes_profile.json").write_text(json.dumps(profile), encoding="utf-8")

def frame(index, energy):
    if no_energy:
        head = "path frame %d" % index
    else:
        head = "path frame %d energy: %.12f xtb: 6.7.1 (fake)" % (index, energy)
    return ("2\\n" + head + "\\n"
            "H 0.00000000 0.00000000 0.00000000\\n"
            "H 0.74000000 0.00000000 0.00000000\\n")

energies = [0.0, 2.5, 1.0]
xyz = "".join(frame(index, energy) for index, energy in enumerate(energies))
for index in range(len(energies)):
    (frames_dir / ("path_frame_%03d.xyz" % index)).write_text(
        frame(index, 0.0), encoding="utf-8")
(result / "xtbpath.xyz").write_text(xyz, encoding="utf-8")

def product(pid, rel):
    target = output / "RESULT" / rel
    data = target.read_bytes()
    return {"id": pid, "label": pid, "path": rel, "kind": "file",
            "metadata": {"sha256": hashlib.sha256(data).hexdigest(),
                         "size_bytes": len(data)}}

products = [product("pes_profile", "pes_search/pes_profile.json"),
            product("raw_trajectory", "pes_search/xtbpath.xyz"),
            product("path_frame_000", "pes_search/path_frames/path_frame_000.xyz")]
manifest = {"version": 2, "task_id": "", "workflow": "XtbPathSearch",
            "status": "completed", "products": products}
if bad_manifest:
    manifest["workflow"] = "PESsearch"
(output / "RESULT" / "result_manifest.json").write_text(
    json.dumps(manifest), encoding="utf-8")
'''


def _fake_acp(tmp_path: Path) -> Path:
    root = tmp_path / "fake_acp"
    package = root / "src" / "acp"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "cli.py").write_text(_FAKE_CLI, encoding="utf-8")
    return root


def _reaction_dir(tmp_path: Path, name: str = "RXN_0000000001") -> Path:
    return tmp_path / "interim" / "g2" / "paths" / "00000" / name


def _helper_config(extra_args: tuple[str, ...] = ()) -> dict:
    return {
        "g2": {
            "path": {"npoint": 50, "alp": 0.5},
            "xtb": {"threads": 2, "seed": 42, "extra_args": list(extra_args)},
        }
    }


def _plan(extra_args: tuple[str, ...] = (), **overrides) -> dict:
    plan = {
        "reaction_id": "RXN_0000000001",
        "start_xyz_text": START_XYZ,
        "end_xyz_text": END_XYZ,
        "charge": 0,
        "multiplicity": 1,
        "recipe": {
            "path": {"npoint": 50, "alp": 0.5},
            "threads": 2,
            "timeout_seconds": 30.0,
            "seed": 42,
            "extra_args": list(extra_args),
        },
        "content_sha256": "ab" * 32,
    }
    plan.update(overrides)
    return plan


def _materials(tmp_path: Path, root: Path, **overrides) -> dict:
    materials = {
        "reaction_dir": _reaction_dir(tmp_path),
        "acp_root": root,
        "python_executable": sys.executable,
        "acp_config_path": None,
    }
    materials.update(overrides)
    return materials


def _run_helper(tmp_path: Path, root: Path, *, extra_args: tuple[str, ...] = (),
                direction: str = DIRECTION_FORWARD) -> XtbPathAttemptOutcome:
    return run_xtb_path_acp_attempt(
        reaction_dir=_reaction_dir(tmp_path),
        config=_helper_config(extra_args),
        direction=direction,
        execution_id="execution-backend-001",
        attempt_id="attempt-backend-001",
        timeout_seconds=10.0,
        charge=0,
        uhf=0,
        start_xyz_text=START_XYZ,
        end_xyz_text=END_XYZ,
        acp_root=root,
        python_executable=sys.executable,
        acp_config_path=None,
    )


# ---------------------------------------------------------------------------
# Per-attempt helper.
# ---------------------------------------------------------------------------
def test_helper_success_parses_frames_and_acp_provenance(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    outcome = _run_helper(tmp_path, root)

    assert isinstance(outcome, XtbPathAttemptOutcome)
    assert outcome.failure_detail is None
    assert outcome.frames is not None
    assert len(outcome.frames) == len(FRAME_ENERGIES)
    for frame, expected_energy in zip(outcome.frames, FRAME_ENERGIES, strict=True):
        assert isinstance(frame, Frame)
        assert frame.energy == pytest.approx(expected_energy)
        assert frame.elements == ("H", "H")
        assert len(frame.coordinates) == 2

    attempt = outcome.attempt
    assert attempt["direction"] == DIRECTION_FORWARD
    assert attempt["returncode"] == 0
    assert attempt["timed_out"] is False
    assert "failure_code" not in attempt
    assert attempt["request_sha256"]
    assert attempt["manifest_sha256"]
    assert attempt["acp_execution_id"] == "execution-backend-001"
    assert attempt["acp_attempt_id"] == "attempt-backend-001"
    assert attempt["wall_seconds"] >= 0.0
    assert attempt["raw_trajectory_path"]
    assert attempt["raw_trajectory_sha256"]
    # g2_path_v1 attempt keys stay None: ACP's CLI transport surfaces none.
    for key in (
        "executable_sha256",
        "argv",
        "xtb_version_line",
        "seed_supported",
        "seed",
        "omp_num_threads",
    ):
        assert key in attempt
        assert attempt[key] is None

    # The request the fake CLI received carried the helper's recipe wiring.
    work = _reaction_dir(tmp_path) / "run" / "WORK" / "pes2ts"
    on_disk = json.loads((work / "path_config.json").read_text(encoding="utf-8"))
    assert on_disk["reaction_id"] == "RXN_0000000001"
    assert on_disk["recipe"]["threads"] == 2
    assert on_disk["recipe"]["uhf"] == 0
    assert on_disk["recipe"]["timeout_seconds"] == 10.0
    assert "npoint=50" in on_disk["recipe"]["path_inp_text"]
    assert on_disk["provenance"]["request_sha256"] == attempt["request_sha256"]
    assert (work / "start.xyz").read_text(encoding="utf-8") == START_XYZ


def test_helper_transport_failure_is_typed_g2_xtb_failed(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    outcome = _run_helper(tmp_path, root, extra_args=("--test-exit-code", "9"))

    assert outcome.frames is None
    assert outcome.failure_detail is not None
    assert "return code 9" in outcome.failure_detail
    assert outcome.attempt["failure_code"] == RejectionCode.G2_XTB_FAILED.value
    assert outcome.attempt["returncode"] == 9
    assert outcome.attempt["acp_status"] == "failed"
    assert outcome.attempt["manifest_sha256"] is None
    assert outcome.attempt["raw_trajectory_sha256"] is None


def test_helper_parse_failure_is_typed_g2_xtb_failed(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    outcome = _run_helper(tmp_path, root, extra_args=("--test-no-energy",))

    # Transport completed (manifest valid) but the raw trajectory is
    # unparseable — the SAME failure class as a malformed local xTB output.
    assert outcome.frames is None
    assert outcome.failure_detail is not None
    assert "unusable ACP xTB path output" in outcome.failure_detail
    assert outcome.attempt["failure_code"] == RejectionCode.G2_XTB_FAILED.value
    assert outcome.attempt["acp_status"] == "completed"


def test_helper_missing_acp_root_raises_value_error(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="acp_root is required"):
        run_xtb_path_acp_attempt(
            reaction_dir=_reaction_dir(tmp_path),
            config=_helper_config(),
            direction=DIRECTION_FORWARD,
            execution_id="execution-backend-001",
            attempt_id="attempt-backend-001",
            timeout_seconds=10.0,
            charge=0,
            uhf=0,
            start_xyz_text=START_XYZ,
            end_xyz_text=END_XYZ,
        )


def _receipt_command(reaction_dir: Path, direction: str = "run") -> list[str]:
    work = reaction_dir / direction / "WORK" / "pes2ts"
    receipt = json.loads((work / "cli_receipt.json").read_text(encoding="utf-8"))
    return receipt["command"]


def test_helper_threads_register_from_config(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    for configured, expected in ((True, True), (False, False)):
        reaction_dir = _reaction_dir(tmp_path, f"RXN_register_{configured}")
        config = _helper_config()
        config["acp"] = {"register": configured}
        run_xtb_path_acp_attempt(
            reaction_dir=reaction_dir, config=config,
            direction=DIRECTION_FORWARD,
            execution_id=f"execution-register-{configured}",
            attempt_id=f"attempt-register-{configured}",
            timeout_seconds=10.0, charge=0, uhf=0,
            start_xyz_text=START_XYZ, end_xyz_text=END_XYZ,
            acp_root=root, python_executable=sys.executable,
        )
        command = _receipt_command(reaction_dir)
        assert ("--register" in command) is expected


def test_backend_run_plumbs_register_from_constructor_and_materials(
    tmp_path: Path,
) -> None:
    root = _fake_acp(tmp_path)
    backend = XtbPathACPBackend(register=False)
    reaction_dir = _reaction_dir(tmp_path, "RXN_backend_register")
    materials = {
        "reaction_dir": reaction_dir, "acp_root": root,
        "python_executable": sys.executable, "acp_config_path": None,
    }
    backend.run(_plan(), materials)
    assert "--register" not in _receipt_command(reaction_dir)

    reaction_dir2 = _reaction_dir(tmp_path, "RXN_backend_register_on")
    backend.run(
        _plan(), {**materials, "reaction_dir": reaction_dir2, "register": True}
    )
    assert "--register" in _receipt_command(reaction_dir2)


# ---------------------------------------------------------------------------
# ExecutionBackend projection.
# ---------------------------------------------------------------------------
def test_backend_satisfies_execution_backend_protocol() -> None:
    backend = XtbPathACPBackend()
    assert backend.method == METHOD_XTB_PATH
    assert isinstance(backend, ExecutionBackend)


def test_backend_run_success_projects_completed_trajectory_record(
    tmp_path: Path,
) -> None:
    root = _fake_acp(tmp_path)
    backend = XtbPathACPBackend()
    plan = _plan()
    materials = _materials(tmp_path, root)

    record = backend.run(plan, materials)

    assert isinstance(record, TrajectoryRecord)
    assert record.reaction_id == "RXN_0000000001"
    assert record.method == METHOD_XTB_PATH
    assert record.status == STATUS_COMPLETED
    assert record.plan_sha256 == "ab" * 32
    assert record.schema_version == "pes2ts_trajectory_record_v1"
    assert len(record.frames) == len(FRAME_ENERGIES)
    for index, (frame, expected_energy) in enumerate(
        zip(record.frames, FRAME_ENERGIES, strict=True)
    ):
        assert frame.frame_index == index
        assert frame.energy_hartree == pytest.approx(expected_energy)
        assert frame.geometry is not None
        assert frame.geometry[1] == pytest.approx((0.74, 0.0, 0.0))

    acp_block = record.provenance["acp"]
    assert acp_block["direction"] == DIRECTION_FORWARD
    assert "failure_code" not in acp_block
    assert acp_block["returncode"] == 0
    assert acp_block["timed_out"] is False
    assert acp_block["acp_execution_id"].startswith("pes2ts-xb-path-")
    assert acp_block["acp_attempt_id"].startswith("attempt-")
    assert acp_block["request_sha256"]
    assert acp_block["manifest_sha256"]
    assert acp_block["raw_trajectory_sha256"]
    assert record.provenance["request_sha256"] == acp_block["request_sha256"]
    assert (
        record.provenance["raw_trajectory_sha256"]
        == acp_block["raw_trajectory_sha256"]
    )


def test_backend_run_failure_never_synthesizes_completed(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    backend = XtbPathACPBackend()
    plan = _plan(extra_args=("--test-exit-code", "9"))
    materials = _materials(tmp_path, root)

    record = backend.run(plan, materials)

    assert record.status == STATUS_FAILED
    assert record.frames == ()
    acp_block = record.provenance["acp"]
    assert acp_block["failure_code"] == RejectionCode.G2_XTB_FAILED.value
    assert record.provenance["request_sha256"] == acp_block["request_sha256"]
    assert record.provenance["raw_trajectory_sha256"] is None


def test_backend_run_parse_failure_is_failed_record(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    backend = XtbPathACPBackend()
    plan = _plan(extra_args=("--test-no-energy",))
    materials = _materials(tmp_path, root)

    record = backend.run(plan, materials)

    assert record.status == STATUS_FAILED
    assert record.provenance["acp"]["failure_code"] == (
        RejectionCode.G2_XTB_FAILED.value
    )


def test_backend_run_requires_reaction_dir_in_materials(tmp_path: Path) -> None:
    root = _fake_acp(tmp_path)
    backend = XtbPathACPBackend()
    with pytest.raises(ValueError, match="reaction_dir"):
        backend.run(_plan(), {"acp_root": root, "python_executable": sys.executable})


# ---------------------------------------------------------------------------
# prepare: plan/materials validation.
# ---------------------------------------------------------------------------
def test_prepare_validates_and_materializes_run_directory(tmp_path: Path) -> None:
    backend = XtbPathACPBackend()
    reaction_dir = _reaction_dir(tmp_path)
    materials = {"reaction_dir": reaction_dir}

    prep = backend.prepare(_plan(), materials)

    assert prep["method"] == METHOD_XTB_PATH
    assert prep["status"] == "ready"
    assert prep["reaction_id"] == "RXN_0000000001"
    assert prep["direction"] == DIRECTION_FORWARD
    assert prep["plan_sha256"] == "ab" * 32
    assert prep["reaction_dir"] == str(reaction_dir)
    assert reaction_dir.is_dir()


def test_prepare_accepts_plan_without_reaction_dir(tmp_path: Path) -> None:
    backend = XtbPathACPBackend()
    prep = backend.prepare(_plan(), {})
    assert prep["reaction_dir"] is None
    assert prep["status"] == "ready"


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda plan: plan.pop("reaction_id"), "plan.reaction_id"),
        (lambda plan: plan.update(reaction_id="  "), "plan.reaction_id"),
        (lambda plan: plan.pop("start_xyz_text"), "plan.start_xyz_text"),
        (lambda plan: plan.update(end_xyz_text=""), "plan.end_xyz_text"),
        (lambda plan: plan.update(charge="0"), "plan.charge"),
        (lambda plan: plan.update(multiplicity=0), "plan.multiplicity"),
        (lambda plan: plan.update(multiplicity=True), "plan.multiplicity"),
        (lambda plan: plan.pop("recipe"), "plan.recipe"),
        (lambda plan: plan.update(recipe=[]), "plan.recipe"),
        (
            lambda plan: plan.update(direction="sideways"),
            "plan.direction",
        ),
        (
            lambda plan: plan.update(recipe={"path": "not-a-mapping"}),
            "plan.recipe.path",
        ),
    ],
)
def test_prepare_rejects_malformed_plans(mutate, match: str) -> None:
    backend = XtbPathACPBackend()
    plan = _plan()
    mutate(plan)
    with pytest.raises(ValueError, match=match):
        backend.prepare(plan, {})


def test_prepare_rejects_non_mapping_plan() -> None:
    backend = XtbPathACPBackend()
    with pytest.raises(ValueError, match="plan must be a mapping"):
        backend.prepare(["not", "a", "plan"], {})  # type: ignore[arg-type]
