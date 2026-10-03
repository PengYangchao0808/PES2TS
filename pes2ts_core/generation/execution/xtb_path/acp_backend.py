"""ACP-executed XTB_PATH backend and per-attempt helper (ADR-0002 X2'-B).

This module is the PES2TS side of the ACP execution convergence: it owns no
chemistry engine.  :func:`run_xtb_path_acp_attempt` is the per-attempt helper
the G2 pipeline calls in place of the deleted local runner (ADR-0002 X2'-C);
it mirrors the ``pipeline._run_attempt`` contract conceptually — ``(frames |
None, failure detail | None, attempt provenance dict)`` — but executes through
the ACP ``XtbPathSearch`` CLI transport and parses ACP's raw trajectory with
the SAME :func:`pes2ts_core.generation.execution.xtb_path.xtb_output.
parse_path_xyz` parser the local pipeline used, preserving ReplayParity.

:class:`XtbPathACPBackend` implements the frozen :class:`ExecutionBackend`
protocol (``method = "XTB_PATH"``) on top of that helper and projects one
:class:`TrajectoryRecord` per call.  Failures are typed
(``RejectionCode.G2_XTB_FAILED``); a ``completed`` status is never synthesized
when the attempt failed or no frames were parsed.

Plan contract consumed by the backend (minimal, X2'-B):

* ``reaction_id`` (non-empty str),
* ``start_xyz_text`` / ``end_xyz_text`` (non-empty str),
* ``charge`` (int), ``multiplicity`` (int >= 1),
* ``recipe`` (mapping): the frozen xTB PATH recipe.  ``recipe["path"]`` is the
  ``g2.path`` ``$path`` parameter section (or the recipe itself may be that
  flat section); optional ``threads`` / ``timeout_seconds`` / ``seed`` /
  ``extra_args`` mirror ``g2.xtb``,
* ``direction`` (optional: ``"forward"`` | ``"reverse"``, default forward),
* ``content_sha256`` (optional): binds ``TrajectoryRecord.plan_sha256``.

Materials contract: ``reaction_dir`` (per-reaction run directory; its name
must be the reaction id), optional ``config`` mapping, and optional ACP
wiring ``acp_root`` / ``python_executable`` / ``acp_config_path``.  Wiring
falls back to the backend constructor and then ``config["acp"]``.

Unit honesty note: the parsed xTB PATH frame energies are the RELATIVE
kcal/mol comment values from ``xtbpath.xyz`` (see ``xtb_output.Frame``).
They are carried into the frozen seam field ``TrajectoryFrame.energy_hartree``
because that field is the seam's only energy slot; the native unit is
recorded explicitly in ``TrajectoryRecord.provenance["acp"]
["native_frame_energy_unit"]`` so downstream projections (X3') can re-map it
without guessing.

This module is truth-free: no ground-truth imports or quarantined-path
strings; the static truth guard scans it like every other G2 module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

from pes2ts_core.g0.rejections import RejectionCode
from pes2ts_core.generation.execution.protocol import METHOD_XTB_PATH
from pes2ts_core.generation.execution.record import (
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_PARTIAL,
    TrajectoryFrame,
    TrajectoryRecord,
)
from pes2ts_core.generation.execution.xtb_path.xtb_output import (
    Frame,
    XtbOutputError,
    parse_path_xyz,
)
from pes2ts_core.integration.acp.xtb_path_request import build_path_request
from pes2ts_core.integration.acp.xtb_path_transport import (
    XtbPathAttemptResult,
    run_xtb_path_attempt,
)
from pes2ts_core.utils.hashing import sha256_bytes, sha256_file, stable_json_dumps

#: Attempt directions, matching the G2 pipeline vocabulary.
DIRECTION_FORWARD: Final[str] = "forward"
DIRECTION_REVERSE: Final[str] = "reverse"
#: Per-reaction ACP attempt directory beneath ``reaction_dir`` (pipeline layout).
RUN_DIRNAME: Final[str] = "run"
REVERSE_RUN_DIRNAME: Final[str] = "run_reverse"
#: PES2TS-owned metadata directory inside the ACP attempt ``WORK`` root.
_PES2TS_WORK_DIRNAME: Final[tuple[str, ...]] = ("WORK", "pes2ts")
#: Start-geometry filename kept beside the ACP request for the start_path check.
_START_XYZ_FILENAME: Final[str] = "start.xyz"
#: Default GFN level of the frozen xTB PATH command contract.
_DEFAULT_GFN_LEVEL: Final[int] = 2
_DEFAULT_THREADS: Final[int] = 4
_DEFAULT_TIMEOUT_SECONDS: Final[float] = 1800.0
#: Native unit of parsed ``xtbpath.xyz`` comment energies.
_NATIVE_FRAME_ENERGY_UNIT: Final[str] = "relative_kcal_per_mol"
#: Keys of the frozen ``g2_path_v1`` attempt block the local runner filled and
#: ACP's CLI transport does not surface; they stay ``None`` (never fabricated).
_G2_ATTEMPT_NULL_KEYS: Final[tuple[str, ...]] = (
    "executable_sha256",
    "argv",
    "xtb_version_line",
    "seed_supported",
    "seed",
    "omp_num_threads",
)


@dataclass(frozen=True, slots=True)
class XtbPathAttemptOutcome:
    """Typed outcome of one ACP ``XtbPathSearch`` attempt for one reaction.

    The shape matches the ``pipeline._run_attempt`` contract conceptually so
    X2'-C can drop this helper into the pipeline with minimal change:
    ``frames`` is ``None`` on any failure, ``failure_detail`` is ``None`` on
    success, and ``attempt`` carries ACP provenance plus the ``g2_path_v1``
    attempt keys (``None`` where ACP does not surface a value).
    """

    frames: tuple[Frame, ...] | None
    failure_detail: str | None
    attempt: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _PlanExecutionFields:
    """Validated backend-facing fields extracted from the plan mapping."""

    reaction_id: str
    start_xyz_text: str
    end_xyz_text: str
    charge: int
    multiplicity: int
    uhf: int
    direction: str
    recipe: Mapping[str, Any]
    path_config: Mapping[str, Any]


def _require_nonempty_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _path_config_from_recipe(recipe: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the ``$path`` parameter section carried by *recipe*."""
    if "path" in recipe:
        nested = recipe["path"]
        if not isinstance(nested, Mapping):
            raise ValueError("plan.recipe.path must be a mapping when present")
        return dict(nested)
    return dict(recipe)


def _plan_execution_fields(plan: Mapping[str, Any]) -> _PlanExecutionFields:
    """Validate the plan and extract the execution fields; never defaults."""
    if not isinstance(plan, Mapping):
        raise ValueError("plan must be a mapping")
    reaction_id = _require_nonempty_text(plan.get("reaction_id"), "plan.reaction_id")
    start_xyz_text = _require_nonempty_text(plan.get("start_xyz_text"), "plan.start_xyz_text")
    end_xyz_text = _require_nonempty_text(plan.get("end_xyz_text"), "plan.end_xyz_text")
    charge = _require_int(plan.get("charge"), "plan.charge")
    multiplicity = _require_int(plan.get("multiplicity"), "plan.multiplicity")
    if multiplicity < 1:
        raise ValueError("plan.multiplicity must be an integer >= 1")
    recipe = plan.get("recipe")
    if not isinstance(recipe, Mapping):
        raise ValueError("plan.recipe must be a mapping")
    direction = plan.get("direction", DIRECTION_FORWARD)
    if direction not in (DIRECTION_FORWARD, DIRECTION_REVERSE):
        raise ValueError(
            f"plan.direction must be {DIRECTION_FORWARD!r} or {DIRECTION_REVERSE!r}, "
            f"got {direction!r}"
        )
    return _PlanExecutionFields(
        reaction_id=reaction_id,
        start_xyz_text=start_xyz_text,
        end_xyz_text=end_xyz_text,
        charge=charge,
        multiplicity=multiplicity,
        uhf=max(0, multiplicity - 1),
        direction=str(direction),
        recipe=recipe,
        path_config=_path_config_from_recipe(recipe),
    )


def _plan_sha256(plan: Mapping[str, Any]) -> str:
    """Return the plan binding digest: sealed ``content_sha256`` when present."""
    sealed = plan.get("content_sha256") if isinstance(plan, Mapping) else None
    if isinstance(sealed, str) and sealed:
        return sealed
    return sha256_bytes(stable_json_dumps(dict(plan)).encode("utf-8"))


def _materials_reaction_dir(materials: Mapping[str, Any] | None) -> Path | None:
    if not isinstance(materials, Mapping):
        return None
    raw = materials.get("reaction_dir")
    if raw is None:
        return None
    return Path(raw)


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    block = config.get(name) if isinstance(config, Mapping) else None
    return block if isinstance(block, Mapping) else {}


def _g2_xtb(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _section(_section(config, "g2"), "xtb")


def _g2_path(config: Mapping[str, Any]) -> Mapping[str, Any]:
    return _section(_section(config, "g2"), "path")


def _resolve_acp_wiring(
    config: Mapping[str, Any],
    acp_root: str | Path | None,
    python_executable: str | Path | None,
    acp_config_path: str | Path | None,
) -> tuple[str | Path | None, str | Path | None, str | Path | None]:
    """Explicit parameters win; ``config["acp"]`` fills the rest."""
    acp_cfg = _section(config, "acp")
    root = acp_root if acp_root is not None else acp_cfg.get("root")
    python = (
        python_executable
        if python_executable is not None
        else acp_cfg.get("python_executable")
    )
    cfg_path = (
        acp_config_path
        if acp_config_path is not None
        else acp_cfg.get("config_path")
    )
    return root, python, cfg_path


def _normalize_extra_args(raw: Any) -> tuple[str, ...]:
    if raw is None:
        return ()
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
        raise ValueError("g2.xtb.extra_args must be a sequence of strings")
    return tuple(_require_nonempty_text(arg, "extra_args item") for arg in raw)


def _run_dir_for(direction: str) -> str:
    if direction == DIRECTION_FORWARD:
        return RUN_DIRNAME
    if direction == DIRECTION_REVERSE:
        return REVERSE_RUN_DIRNAME
    raise ValueError(
        f"unknown direction {direction!r}; expected {DIRECTION_FORWARD!r} "
        f"or {DIRECTION_REVERSE!r}"
    )


def _project_attempt_result(
    result: XtbPathAttemptResult,
    *,
    direction: str,
    start_xyz_text: str,
) -> XtbPathAttemptOutcome:
    """Map one transport result to frames + typed detail + provenance dict."""
    attempt: dict[str, Any] = {
        "direction": direction,
        "returncode": result.returncode,
        "timed_out": result.timed_out,
        "request_sha256": result.request_sha256,
        "manifest_sha256": None,
        "acp_execution_id": result.execution_id,
        "acp_attempt_id": result.attempt_id,
        "wall_seconds": result.wall_seconds,
        "acp_status": result.status,
        "acp_reused": result.reused,
        "acp_attempt_dir": result.attempt_dir,
        "acp_log_ref": result.log_ref,
        "acp_error": result.error,
        "native_frame_energy_unit": _NATIVE_FRAME_ENERGY_UNIT,
        "raw_trajectory_path": None,
        "raw_trajectory_sha256": None,
    }
    for key in _G2_ATTEMPT_NULL_KEYS:
        attempt[key] = None
    if result.status != "completed":
        detail = result.error or (
            f"ACP XtbPathSearch attempt status is {result.status!r}"
        )
        attempt["failure_code"] = RejectionCode.G2_XTB_FAILED.value
        return XtbPathAttemptOutcome(frames=None, failure_detail=str(detail), attempt=attempt)
    if not result.raw_trajectory_path:
        detail = "ACP completed attempt did not record a raw trajectory path"
        attempt["failure_code"] = RejectionCode.G2_XTB_FAILED.value
        return XtbPathAttemptOutcome(frames=None, failure_detail=detail, attempt=attempt)
    raw_trajectory = Path(result.raw_trajectory_path)
    if not raw_trajectory.is_file():
        detail = (
            "ACP completed attempt is missing its raw trajectory at "
            f"{result.raw_trajectory_path}"
        )
        attempt["failure_code"] = RejectionCode.G2_XTB_FAILED.value
        return XtbPathAttemptOutcome(frames=None, failure_detail=detail, attempt=attempt)
    start_path = Path(result.attempt_dir).joinpath(*_PES2TS_WORK_DIRNAME, _START_XYZ_FILENAME)
    try:
        start_path.parent.mkdir(parents=True, exist_ok=True)
        start_path.write_text(start_xyz_text, encoding="utf-8")
        frames = parse_path_xyz(raw_trajectory, start_path=start_path)
        manifest_sha256 = (
            sha256_file(result.manifest_path) if result.manifest_path else None
        )
        raw_trajectory_sha256 = sha256_file(raw_trajectory)
    except (XtbOutputError, OSError) as error:
        detail = f"unusable ACP xTB path output: {error}"
        attempt["failure_code"] = RejectionCode.G2_XTB_FAILED.value
        return XtbPathAttemptOutcome(frames=None, failure_detail=detail, attempt=attempt)
    attempt["manifest_sha256"] = manifest_sha256
    attempt["raw_trajectory_path"] = str(raw_trajectory)
    attempt["raw_trajectory_sha256"] = raw_trajectory_sha256
    return XtbPathAttemptOutcome(frames=frames, failure_detail=None, attempt=attempt)


def run_xtb_path_acp_attempt(
    *,
    reaction_dir: Path,
    config: Mapping[str, Any],
    direction: str,
    execution_id: str,
    attempt_id: str,
    timeout_seconds: float,
    charge: int,
    uhf: int,
    start_xyz_text: str,
    end_xyz_text: str,
    acp_root: str | Path | None = None,
    python_executable: str | Path | None = None,
    acp_config_path: str | Path | None = None,
    register: bool | None = None,
) -> XtbPathAttemptOutcome:
    """Run one ACP ``XtbPathSearch`` attempt and parse its raw trajectory.

    Builds the frozen ``pes2ts_xtb_path_request_v1`` payload from *config*
    (``g2.path`` as the ``$path`` section, ``g2.xtb.threads`` as threads, the
    configured ``g2.xtb.seed``/``g2.xtb.extra_args`` when present) plus the
    given xyz texts / charge / ``uhf`` (mapped to ``multiplicity = uhf + 1``),
    launches it through :func:`run_xtb_path_attempt` under
    ``reaction_dir / run[_reverse]``, and parses
    ``RESULT/pes_search/xtbpath.xyz`` with :func:`parse_path_xyz` (the same
    parser the local pipeline used).  ``reaction_id`` comes from
    ``config["reaction_id"]`` when set, else ``reaction_dir.name``.

    *register* (explicit argument wins, else ``config["acp"]["register"]``,
    else ``true``) appends ``--register`` to the ACP CLI argv so completed
    runs register in the ACP jobs store and resolve in the Workbench; a
    registration failure fails the attempt with a typed error (see
    :func:`run_xtb_path_attempt`).

    Infrastructure problems (missing ``acp_root``, malformed recipe) raise
    ``ValueError``; attempt-level failures return a typed outcome with
    ``frames=None``, a ``failure_detail`` string, and
    ``attempt["failure_code"] == RejectionCode.G2_XTB_FAILED.value``.
    """
    reaction_dir = Path(reaction_dir)
    direction = str(direction)
    _run_dir_for(direction)  # validates direction early
    if not isinstance(config, Mapping):
        raise ValueError("config must be a mapping")
    g2 = _section(config, "g2")
    path_config = g2.get("path") if isinstance(g2.get("path"), Mapping) else {}
    xtb_cfg = _g2_xtb(config)
    threads = int(xtb_cfg.get("threads", _DEFAULT_THREADS))
    if threads < 1:
        raise ValueError("g2.xtb.threads must be >= 1")
    seed_raw = xtb_cfg.get("seed")
    seed = None if seed_raw is None else _require_int(seed_raw, "g2.xtb.seed")
    extra_args = _normalize_extra_args(xtb_cfg.get("extra_args"))
    configured_reaction_id = config.get("reaction_id")
    reaction_id = (
        str(configured_reaction_id).strip()
        if configured_reaction_id
        else reaction_dir.name
    )
    if not reaction_id or reaction_id in {".", ".."}:
        raise ValueError(
            "reaction_id must come from config['reaction_id'] or reaction_dir.name"
        )
    acp_root, python_executable, acp_config_path = _resolve_acp_wiring(
        config, acp_root, python_executable, acp_config_path
    )
    if acp_root is None:
        raise ValueError(
            "acp_root is required: pass it explicitly or set config['acp']['root']"
        )
    if register is None:
        register_cfg = _section(config, "acp").get("register")
        register = True if register_cfg is None else bool(register_cfg)
    request = build_path_request(
        reaction_id=reaction_id,
        start_xyz_text=start_xyz_text,
        end_xyz_text=end_xyz_text,
        charge=_require_int(charge, "charge"),
        multiplicity=int(uhf) + 1,
        path_config=path_config,
        gfn_level=_DEFAULT_GFN_LEVEL,
        threads=threads,
        timeout_seconds=timeout_seconds,
        seed=seed,
        extra_args=extra_args,
    )
    result = run_xtb_path_attempt(
        acp_root=acp_root,
        python_executable=python_executable,
        acp_config_path=acp_config_path,
        request=request,
        output_root=reaction_dir / _run_dir_for(direction),
        execution_id=execution_id,
        attempt_id=attempt_id,
        timeout_seconds=timeout_seconds,
        register=bool(register),
    )
    return _project_attempt_result(
        result, direction=direction, start_xyz_text=start_xyz_text
    )


def _project_frames(
    frames: tuple[Frame, ...] | None,
) -> tuple[tuple[TrajectoryFrame, ...], str]:
    """Map parsed frames to seam frames plus a never-synthesized status."""
    if frames is None:
        return (), STATUS_FAILED
    trajectory_frames = tuple(
        TrajectoryFrame(
            frame_index=index,
            energy_hartree=frame.energy,
            geometry=frame.coordinates,
        )
        for index, frame in enumerate(frames)
    )
    if not trajectory_frames:
        return (), STATUS_PARTIAL
    return trajectory_frames, STATUS_COMPLETED


def _lookup_xtb_setting(
    configs: Sequence[Mapping[str, Any]], key: str, default: Any
) -> Any:
    for config in configs:
        value = _g2_xtb(config).get(key)
        if value is not None:
            return value
    return default


class XtbPathACPBackend:
    """``ExecutionBackend`` for ``XTB_PATH`` executed through ACP (ADR-0002).

    ``prepare`` validates the plan/materials contract and materializes the
    reaction run directory; ``run`` calls :func:`run_xtb_path_acp_attempt`
    with fresh ``execution_id``/``attempt_id`` values and projects the ACP
    outcome into one :class:`TrajectoryRecord`.  ACP wiring may be injected
    via the constructor and overridden per call through ``materials``.
    """

    method: str = METHOD_XTB_PATH

    def __init__(
        self,
        *,
        config: Mapping[str, Any] | None = None,
        acp_root: str | Path | None = None,
        python_executable: str | Path | None = None,
        acp_config_path: str | Path | None = None,
        register: bool | None = None,
    ) -> None:
        self._config: dict[str, Any] = dict(config) if isinstance(config, Mapping) else {}
        self._acp_root = acp_root
        self._python_executable = python_executable
        self._acp_config_path = acp_config_path
        self._register = register

    def prepare(
        self, plan: Mapping[str, object], materials: Mapping[str, object]
    ) -> Mapping[str, object]:
        """Validate the plan and materialize the reaction run directory."""
        fields = _plan_execution_fields(plan)
        reaction_dir = _materials_reaction_dir(materials)
        if reaction_dir is not None:
            reaction_dir.mkdir(parents=True, exist_ok=True)
        return {
            "method": METHOD_XTB_PATH,
            "status": "ready",
            "reaction_id": fields.reaction_id,
            "direction": fields.direction,
            "plan_sha256": _plan_sha256(plan),
            "reaction_dir": None if reaction_dir is None else str(reaction_dir),
        }

    def run(
        self, plan: Mapping[str, object], materials: Mapping[str, object]
    ) -> TrajectoryRecord:
        """Execute one ACP attempt and project it into a TrajectoryRecord."""
        if not isinstance(materials, Mapping):
            raise ValueError("materials must be a mapping")
        fields = _plan_execution_fields(plan)
        reaction_dir = _materials_reaction_dir(materials)
        if reaction_dir is None:
            raise ValueError(
                "materials must carry reaction_dir for an XTB_PATH ACP run"
            )
        base_config = self._config
        material_config = materials.get("config")
        if not isinstance(material_config, Mapping):
            material_config = {}
        configs = (material_config, base_config)
        recipe = fields.recipe
        threads = int(
            recipe.get("threads")
            if recipe.get("threads") is not None
            else _lookup_xtb_setting(configs, "threads", _DEFAULT_THREADS)
        )
        timeout_raw = (
            recipe.get("timeout_seconds")
            if recipe.get("timeout_seconds") is not None
            else _lookup_xtb_setting(configs, "timeout_seconds", _DEFAULT_TIMEOUT_SECONDS)
        )
        timeout_seconds = float(timeout_raw)
        if "seed" in recipe:
            seed_raw = recipe["seed"]
        else:
            seed_raw = _lookup_xtb_setting(configs, "seed", None)
        seed = None if seed_raw is None else _require_int(seed_raw, "recipe.seed")
        if "extra_args" in recipe:
            extra_args = _normalize_extra_args(recipe["extra_args"])
        else:
            extra_args = _normalize_extra_args(
                _lookup_xtb_setting(configs, "extra_args", ())
            )
        acp_root = materials.get("acp_root")
        if acp_root is None:
            acp_root = self._acp_root
        python_executable = materials.get("python_executable")
        if python_executable is None:
            python_executable = self._python_executable
        acp_config_path = materials.get("acp_config_path")
        if acp_config_path is None:
            acp_config_path = self._acp_config_path
        register = materials.get("register")
        if register is None:
            register = self._register
        helper_config = self._helper_config(
            base_config,
            material_config,
            path_config=fields.path_config,
            threads=threads,
            timeout_seconds=timeout_seconds,
            seed=seed,
            extra_args=extra_args,
            reaction_id=fields.reaction_id,
            acp_root=acp_root,
            python_executable=python_executable,
            acp_config_path=acp_config_path,
        )
        outcome = run_xtb_path_acp_attempt(
            reaction_dir=reaction_dir,
            config=helper_config,
            direction=fields.direction,
            execution_id=f"pes2ts-xb-path-{uuid4().hex}",
            attempt_id=f"attempt-{uuid4().hex}",
            timeout_seconds=timeout_seconds,
            charge=fields.charge,
            uhf=fields.uhf,
            start_xyz_text=fields.start_xyz_text,
            end_xyz_text=fields.end_xyz_text,
            acp_root=acp_root,
            python_executable=python_executable,
            acp_config_path=acp_config_path,
            register=register,
        )
        frames, status = _project_frames(outcome.frames)
        acp_block = dict(outcome.attempt)
        provenance: dict[str, Any] = {
            "acp": acp_block,
            "request_sha256": outcome.attempt.get("request_sha256"),
            "raw_trajectory_sha256": outcome.attempt.get("raw_trajectory_sha256"),
        }
        return TrajectoryRecord(
            reaction_id=fields.reaction_id,
            method=METHOD_XTB_PATH,
            status=status,
            frames=frames,
            plan_sha256=_plan_sha256(plan),
            provenance=provenance,
        )

    @staticmethod
    def _helper_config(
        base_config: Mapping[str, Any],
        material_config: Mapping[str, Any],
        *,
        path_config: Mapping[str, Any],
        threads: int,
        timeout_seconds: float,
        seed: int | None,
        extra_args: tuple[str, ...],
        reaction_id: str,
        acp_root: str | Path | None,
        python_executable: str | Path | None,
        acp_config_path: str | Path | None,
    ) -> dict[str, Any]:
        """Assemble the helper-facing config with explicit recipe values."""
        merged: dict[str, Any] = {**base_config, **material_config}
        g2_block = dict(_section(merged, "g2"))
        xtb_block = dict(_g2_xtb(merged))
        xtb_block.update(
            {
                "threads": threads,
                "timeout_seconds": timeout_seconds,
                "seed": seed,
                "extra_args": list(extra_args),
            }
        )
        g2_block["path"] = dict(path_config)
        g2_block["xtb"] = xtb_block
        acp_block = dict(_section(merged, "acp"))
        if acp_root is not None:
            acp_block["root"] = str(acp_root)
        if python_executable is not None:
            acp_block["python_executable"] = str(python_executable)
        if acp_config_path is not None:
            acp_block["config_path"] = str(acp_config_path)
        return {**merged, "g2": g2_block, "acp": acp_block, "reaction_id": reaction_id}


__all__ = [
    "DIRECTION_FORWARD",
    "DIRECTION_REVERSE",
    "REVERSE_RUN_DIRNAME",
    "RUN_DIRNAME",
    "XtbPathACPBackend",
    "XtbPathAttemptOutcome",
    "run_xtb_path_acp_attempt",
]
