"""G2 run orchestration: selection, prepare, run, reverse retry, ledger (task 9).

:func:`select_reaction_ids` intersects the ``g2_eligible`` list with the cohort
/limit/reaction filters (same semantics as ``g1.build.cohort_member_ids``,
independently implemented).  :func:`prepare_ids` assembles and persists the
deterministic endpoints per reaction; every typed failure appends one
``stage="g2_path"`` ledger record with an idempotent signature.
:func:`run_ids` resumes around terminal ``reaction_path.json`` documents
(idempotency = no new ACP calls and unchanged sha256 of ``reaction_path.json``
/``frames.parquet``/the terminal ACP artifacts), runs the xTB PATH step
EXECUTED THROUGH ACP (``acp_backend.run_xtb_path_acp_attempt``; ADR-0002 —
the local xTB runner is deleted), retries start/end-reversed from
``run_reverse/`` when the forward verdict code is in the configured trigger
set, renormalizes a recovered reverse path (reverse frame order, recalibrate
``energy_rel_kcal`` against the new frame 0, keep ``energy_rel_kcal_raw``),
and closes the batch with the summary, manifest (``run`` counter block), and
coverage artifacts.  ACP wiring (``acp.root``/``acp.python``/``acp.config_path``)
is mandatory: a missing ``acp.root`` raises :class:`InfrastructureError` —
there is no silent local fallback.

Per-reaction layout under ``<paths.interim>/g2/paths/<shard>/<id>/``:
``endpoints.json``, ``R.xyz``, ``P.xyz``, ``run/`` (the ACP attempt root:
``WORK/pes2ts/`` request/receipt/log plus ``RESULT/pes_search/`` raw
trajectory, profile, and manifest), ``run_reverse/`` (reversed attempt),
``frames.parquet``, ``reaction_path.json``.
"""

# allow: SIZE_OK -- the plan freezes one pipeline module owning selection,
# prepare, the xTB run loop with reverse retry and direction normalization,
# resume, ledger idempotency, and the batch artifacts; the natural split seam
# (batch writers vs per-reaction engine) is deferred to keep the frozen layout.

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

from pes2ts_core.g0.inventory import INVENTORY_PARQUET_FILENAME
from pes2ts_core.g0.rejections import (
    LEDGER_FILENAME,
    Rejection,
    RejectionCode,
    RejectionLedger,
)
from pes2ts_core.g0.strata import COHORT_STRATIFIED_FILENAME, COHORT_TRIAL_FILENAME
from pes2ts_core.g1.build import reaction_change_path, shard_name
from pes2ts_core.generation.execution.xtb_path import G2_DIRNAME, G2_PATHS_DIRNAME, SCHEMA_PATH
from pes2ts_core.generation.execution.xtb_path.acp_backend import run_xtb_path_acp_attempt
from pes2ts_core.generation.execution.xtb_path.artifacts import (
    build_summary,
    config_digest,
    reaction_document,
    write_coverage,
    write_frames_parquet,
    write_manifest,
    write_summary,
)
from pes2ts_core.generation.assembly.endpoints import (
    EndpointError,
    assemble_endpoints,
    side_bond_pairs,
    write_endpoint_files,
)
from pes2ts_core.generation.execution.xtb_path.inspect import Metric, Verdict, evaluate_validity, frame_metrics
from pes2ts_core.generation.execution.xtb_path.status import (
    DEFAULT_REVERSE_RETRY_TRIGGERS,
    G2_STATUSES,
    STATUS_VALID,
)
from pes2ts_core.generation.execution.xtb_path.xtb_output import (
    Frame,
    XtbOutputError,
    parse_start_xyz,
)
from pes2ts_core.integration.acp.xtb_path_request import build_path_inp_text
from pes2ts_core.utils.hashing import sha256_file
from pes2ts_core.utils.jsonio import read_json, write_json
from pes2ts_core.utils.parquet_io import read_parquet

logger = logging.getLogger(__name__)


class InfrastructureError(RuntimeError):
    """A missing input, artifact, or ACP wiring (mapped to exit 24 at the CLI)."""


STAGE: Final[str] = "g2_path"
ELIGIBLE_FILENAME: Final[str] = "g2_eligible.json"
ELIGIBLE_HINT: Final[str] = "run `g1 gate` first"
INVENTORY_HINT: Final[str] = "run `g0 inventory` first"
COHORT_HINT: Final[str] = "run `g0 cohorts` first"
PREPARE_HINT: Final[str] = "run `g2 prepare` first"
SUMMARY_FILENAME: Final[str] = "g2_summary.parquet"
MANIFEST_FILENAME: Final[str] = "g2_path_manifest.json"
COVERAGE_FILENAME: Final[str] = "g2_coverage.json"
ENDPOINTS_FILENAME: Final[str] = "endpoints.json"
REACTANT_XYZ_FILENAME: Final[str] = "R.xyz"
PRODUCT_XYZ_FILENAME: Final[str] = "P.xyz"
RUN_DIRNAME: Final[str] = "run"
REVERSE_RUN_DIRNAME: Final[str] = "run_reverse"
FRAMES_FILENAME: Final[str] = "frames.parquet"
DOCUMENT_FILENAME: Final[str] = "reaction_path.json"
DIRECTION_FORWARD: Final[str] = "forward"
DIRECTION_REVERSE: Final[str] = "reverse"
#: CLI cohort mode -> ``g0 cohorts`` artifact filename.
COHORT_FILENAMES: Final[dict[str, str]] = {
    "trial": COHORT_TRIAL_FILENAME, "stratified": COHORT_STRATIFIED_FILENAME,
}
DEFAULT_SHARD_SIZE: Final[int] = 1000
#: ACP attempt artifacts digested into the document ``sources`` block; keys
#: are posix paths relative to the terminal run dir, which ``g2 verify``
#: resolves under ``run``/``run_reverse`` exactly like the old flat
#: run-directory filenames.  Only files that actually exist are recorded.
_ACP_SOURCE_KEYS: Final[tuple[str, ...]] = (
    "WORK/pes2ts/path_config.json",
    "RESULT/result_manifest.json",
    "RESULT/pes_search/pes_profile.json",
    "RESULT/pes_search/xtbpath.xyz",
)
#: PES2TS-owned ACP attempt-slot files cleared before a fresh attempt: the
#: ACP CLI transport keeps ONE immutable attempt receipt per output root, and
#: every pipeline attempt is fresh (resume short-circuits on the terminal
#: document), so stale slot state from a prior ``--force``/retry pass must go.
_ACP_SLOT_FILES: Final[tuple[str, ...]] = ("cli_attempt.claim", "cli_receipt.json")
_DEFAULT_XTB_TIMEOUT_SECONDS: Final[float] = 1800.0

LedgerFailure = tuple[str, str, str, str]


@dataclass(frozen=True, slots=True)
class PrepareReport:
    """Outcome of one :func:`prepare_ids` batch."""

    n_selected: int
    n_attempted: int
    n_skipped: int


@dataclass(frozen=True, slots=True)
class RunReport:
    """Outcome of one :func:`run_ids` batch (manifest ``run`` block)."""

    n_selected: int
    n_attempted: int
    n_skipped: int


@dataclass(frozen=True, slots=True)
class _ReactionContext:
    """Everything one per-reaction xTB run needs, loaded once."""

    reaction_id: str
    rxn_dir: Path
    config: Mapping[str, Any]
    charge: int
    uhf: int
    maps: tuple[int, ...]
    elements: dict[int, str]
    reactant_coords: dict[int, tuple[float, float, float]]
    product_coords: dict[int, tuple[float, float, float]]
    r_pairs: frozenset[tuple[int, int]]
    p_pairs: frozenset[tuple[int, int]]
    events: Mapping[str, Sequence[Mapping[str, Any]]]
    g1_path: Path
    endpoint_record: Mapping[str, Any]


def _paths(config: Mapping[str, Any]) -> tuple[Path, Path]:
    return Path(config["paths"]["interim"]), Path(config["paths"]["manifests"])


def _shard_size(config: Mapping[str, Any]) -> int:
    g2 = config.get("g2", {})
    return int(g2.get("shard_size", DEFAULT_SHARD_SIZE)) if isinstance(g2, Mapping) else DEFAULT_SHARD_SIZE


def _rxn_dir(interim_dir: Path, reaction_id: str, shard_size: int) -> Path:
    return interim_dir / G2_DIRNAME / G2_PATHS_DIRNAME / shard_name(reaction_id, shard_size) / reaction_id


def _g2_settings(config: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    g2 = config.get("g2", {})
    block = g2.get(section, {}) if isinstance(g2, Mapping) else {}
    return block if isinstance(block, Mapping) else {}


def _load_eligible(config: Mapping[str, Any]) -> frozenset[str]:
    interim_dir, _ = _paths(config)
    configured = (config.get("g2") or {}).get("eligible_path")
    path = Path(str(configured)) if configured else interim_dir / ELIGIBLE_FILENAME
    if not path.is_file():
        raise InfrastructureError(f"Missing eligible list {path}; {ELIGIBLE_HINT}")
    try:
        document = read_json(path)
    except ValueError as error:
        raise ValueError(f"eligible artifact {path} is not valid JSON: {error}") from None
    raw = document.get("reaction_ids") if isinstance(document, Mapping) else None
    if not isinstance(raw, list):
        raise ValueError(f"eligible artifact {path} has no reaction_ids list")
    return frozenset(str(value) for value in raw)


def _load_inventory(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    interim_dir, _ = _paths(config)
    path = interim_dir / INVENTORY_PARQUET_FILENAME
    if not path.is_file():
        raise InfrastructureError(f"Missing inventory {path}; {INVENTORY_HINT}")
    return {str(row["reaction_id"]): row for row in read_parquet(path).to_pylist()}


def select_reaction_ids(
    config: Mapping[str, Any],
    *,
    cohort: str = "trial",
    limit: int | None = None,
    reactions: Sequence[str] = (),
) -> list[str]:
    """Return the sorted eligible ids after cohort/limit/reaction filtering.

    Mirrors ``g1.build.cohort_member_ids`` semantics without importing its
    selection: ``trial``/``stratified`` read the matching ``g0 cohorts``
    artifact, ``all`` reads every inventory row; an explicit *reactions*
    subset further restricts the result and *limit* truncates the sorted
    intersection with the eligible list.
    """
    interim_dir, _ = _paths(config)
    eligible = _load_eligible(config)
    if cohort == "all":
        members = sorted(_load_inventory(config))
    elif cohort in COHORT_FILENAMES:
        path = interim_dir / COHORT_FILENAMES[cohort]
        if not path.is_file():
            raise InfrastructureError(f"Missing cohort {path}; {COHORT_HINT}")
        try:
            document = read_json(path)
        except ValueError as error:
            raise ValueError(f"cohort artifact {path} is not valid JSON: {error}") from None
        raw = document.get("members") if isinstance(document, Mapping) else None
        if not isinstance(raw, list):
            raise ValueError(f"cohort artifact {path} has no members list")
        members = sorted(str(member) for member in raw)
    else:
        raise ValueError(f"unknown cohort: {cohort!r}")
    if reactions:
        wanted = {str(reaction_id) for reaction_id in reactions}
        members = [member for member in members if member in wanted]
    selected = sorted(set(members) & eligible)
    return selected[:limit] if limit is not None else selected


def _ledger_records(manifests_dir: Path) -> list[dict[str, str]]:
    path = manifests_dir / LEDGER_FILENAME
    if not path.is_file():
        return []
    records: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if isinstance(record, dict):
            records.append(record)
    return records


def _append_ledger(manifests_dir: Path, failures: Sequence[LedgerFailure]) -> None:
    """Append one typed record per failure, skipping existing signatures."""
    if not failures:
        return
    existing = {
        (str(record.get("reaction_id")), str(record.get("stage")),
         str(record.get("code")), str(record.get("detail")))
        for record in _ledger_records(manifests_dir)
    }
    ledger = RejectionLedger.load(manifests_dir)
    for reaction_id, code, detail, source_pointer in failures:
        if (reaction_id, STAGE, code, detail) in existing:
            continue
        ledger.add(Rejection(
            reaction_id=reaction_id, stage=STAGE, code=RejectionCode(code),
            detail=detail, source_pointer=source_pointer,
        ))
    ledger.write()


def _g1_check(
    interim_dir: Path, reaction_id: str, shard_size: int
) -> tuple[tuple[Mapping[str, Any], Path] | None, LedgerFailure | None]:
    """Return the valid G1 document or a typed ledger failure for it."""
    doc_path = reaction_change_path(interim_dir, reaction_id, shard_size)
    if not doc_path.is_file():
        detail = f"missing G1 document {doc_path}"
        return None, (reaction_id, RejectionCode.G2_MISSING_G1_DOC.value, detail, str(doc_path))
    document = read_json(doc_path)
    validation = document.get("validation") if isinstance(document, Mapping) else None
    status = validation.get("status") if isinstance(validation, Mapping) else None
    if status != "valid":
        detail = f"G1 validation status is {status!r}"
        return None, (reaction_id, RejectionCode.G2_G1_NOT_VALID.value, detail, str(doc_path))
    return (document, doc_path), None


def _prepare_one(
    reaction_id: str,
    *,
    config: Mapping[str, Any],
    interim_dir: Path,
    shard_size: int,
    eligible: frozenset[str],
    rows_by_id: Mapping[str, Mapping[str, Any]],
) -> LedgerFailure | None:
    if reaction_id not in eligible:
        detail = "reaction is not in the eligible list"
        return (reaction_id, RejectionCode.G2_NOT_ELIGIBLE.value, detail, ELIGIBLE_FILENAME)
    g1, failure = _g1_check(interim_dir, reaction_id, shard_size)
    if failure is not None:
        return failure
    document, doc_path = g1
    row = rows_by_id.get(reaction_id)
    if row is None:
        raise InfrastructureError(f"missing inventory row for {reaction_id}; {INVENTORY_HINT}")
    try:
        assembly = assemble_endpoints(document, row, config=config)
    except EndpointError as error:
        return (reaction_id, error.code.value, error.detail, str(doc_path))
    write_endpoint_files(_rxn_dir(interim_dir, reaction_id, shard_size), assembly)
    return None


def prepare_ids(ids: Sequence[str], *, config: Mapping[str, Any]) -> PrepareReport:
    """Assemble and persist the endpoints for every selected reaction.

    Per reaction: eligibility check, G1 document load and validity gate, then
    ``assemble_endpoints`` + ``write_endpoint_files``.  Every typed failure
    appends one idempotent ``stage="g2_path"`` ledger record.
    """
    interim_dir, manifests_dir = _paths(config)
    shard_size = _shard_size(config)
    eligible = _load_eligible(config)
    rows_by_id = _load_inventory(config)
    selected = sorted(dict.fromkeys(str(reaction_id) for reaction_id in ids))
    failures: list[LedgerFailure] = []
    prepared = 0
    for reaction_id in selected:
        failure = _prepare_one(
            reaction_id, config=config, interim_dir=interim_dir, shard_size=shard_size,
            eligible=eligible, rows_by_id=rows_by_id,
        )
        if failure is None:
            prepared += 1
        else:
            failures.append(failure)
    _append_ledger(manifests_dir, failures)
    logger.info("G2 prepare: %d selected, %d prepared, %d failed", len(selected), prepared, len(failures))
    return PrepareReport(len(selected), prepared, len(selected) - prepared)


def _terminal_document(path: Path) -> Mapping[str, Any] | None:
    """Return the persisted terminal document at *path*, or None."""
    if not path.is_file():
        return None
    try:
        document = read_json(path)
    except ValueError:
        return None
    if not isinstance(document, Mapping):
        return None
    if document.get("schema_version") != SCHEMA_PATH or document.get("status") not in G2_STATUSES:
        return None
    return document


def _reaction_context(
    reaction_id: str,
    rxn_dir: Path,
    document: Mapping[str, Any],
    row: Mapping[str, Any],
    config: Mapping[str, Any],
    g1_path: Path,
) -> _ReactionContext:
    maps = tuple(sorted({int(item["map"]) for block in document["reactants"] for item in block["rows"]}))
    endpoints_path = rxn_dir / ENDPOINTS_FILENAME
    if not endpoints_path.is_file():
        raise XtbOutputError(f"missing {ENDPOINTS_FILENAME}")
    try:
        reactant = parse_start_xyz(rxn_dir / REACTANT_XYZ_FILENAME)
        product = parse_start_xyz(rxn_dir / PRODUCT_XYZ_FILENAME)
        endpoint_record = read_json(endpoints_path)
    except XtbOutputError as error:
        raise XtbOutputError(f"{reaction_id}: unreadable prepared endpoints: {error}") from None
    if len(reactant.elements) != len(maps) or len(product.elements) != len(maps):
        raise XtbOutputError(
            f"{reaction_id}: endpoint files hold {len(reactant.elements)}/{len(product.elements)} "
            f"atoms, expected {len(maps)}"
        )
    r_pairs, p_pairs = side_bond_pairs(str(row["reaction_smiles"]))
    return _ReactionContext(
        reaction_id=reaction_id,
        rxn_dir=rxn_dir,
        config=config,
        charge=int(row.get("charge_total_reactants", 0)),
        uhf=max(0, int(row.get("multiplicity_max", 1)) - 1),
        maps=maps,
        elements=dict(zip(maps, reactant.elements)),
        reactant_coords=dict(zip(maps, reactant.coordinates)),
        product_coords=dict(zip(maps, product.coordinates)),
        r_pairs=r_pairs,
        p_pairs=p_pairs,
        events=document.get("bond_changes") or {},
        g1_path=g1_path,
        endpoint_record=endpoint_record if isinstance(endpoint_record, Mapping) else {},
    )


def _judge(frames: tuple[Frame, ...] | None, xtb_failure: str | None, ctx: _ReactionContext) -> Verdict:
    """Verdict of an R→P-ordered frame sequence (xTB failure short-circuits)."""
    return evaluate_validity(
        frames or (),
        reaction_id=ctx.reaction_id,
        reactant_coords=ctx.reactant_coords,
        product_coords=ctx.product_coords,
        elements=ctx.elements,
        r_pairs=ctx.r_pairs,
        p_pairs=ctx.p_pairs,
        events=ctx.events,
        xtb_failure=xtb_failure,
        config=ctx.config,
    )


def _acp_wiring(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the ACP wiring kwargs for one attempt.

    ``acp.root`` (the ACP checkout) is mandatory — an empty/missing value
    raises :class:`InfrastructureError`; G2 executes xTB PATH through ACP
    only, with no silent local fallback.  ``acp.python`` (the ACP environment
    interpreter) and ``acp.config_path`` are optional.  ``acp.register``
    (default ``true``) decides whether the CLI run is registered into the ACP
    jobs store so the Workbench can serve it; a registration failure fails
    the attempt loudly (see ``xtb_path_transport.run_xtb_path_attempt``).
    """
    acp = config.get("acp")
    block = acp if isinstance(acp, Mapping) else {}
    root = block.get("root")
    if not root or not str(root).strip():
        raise InfrastructureError(
            "acp.root is not configured; set acp.root to the ACP checkout "
            "(e.g. .../ACP_V1_20260811) — G2 xTB PATH execution goes through "
            "ACP only; there is no local fallback"
        )
    python = block.get("python")
    config_path = block.get("config_path")
    register_raw = block.get("register")
    return {
        "acp_root": str(root),
        "python_executable": None if python in (None, "") else str(python),
        "acp_config_path": None if config_path in (None, "") else str(config_path),
        "register": True if register_raw is None else bool(register_raw),
    }


def _reset_acp_attempt_state(run_dir: Path) -> None:
    """Clear a previous ACP attempt slot so a fresh attempt may be claimed.

    The ACP CLI transport keeps ONE immutable attempt per output root
    (``WORK/pes2ts/cli_receipt.json`` + claim).  The pipeline always executes
    a fresh attempt whenever it reaches :func:`_run_attempt` (resume
    short-circuits on the terminal document), so stale slot state left by a
    prior ``--force`` rerun or a reverse-retry pass is cleared first — the ACP
    analogue of the local runner overwriting its own run-directory outputs.
    """
    work = run_dir / "WORK" / "pes2ts"
    if not work.is_dir():
        return
    for name in _ACP_SLOT_FILES:
        (work / name).unlink(missing_ok=True)


def _run_attempt(
    ctx: _ReactionContext, run_dirname: str, direction: str
) -> tuple[tuple[Frame, ...] | None, str | None, dict[str, Any]]:
    """One ACP ``XtbPathSearch`` attempt; returns (frames, failure, attempt)."""
    if direction == DIRECTION_FORWARD:
        start_filename, end_filename = REACTANT_XYZ_FILENAME, PRODUCT_XYZ_FILENAME
    else:
        start_filename, end_filename = PRODUCT_XYZ_FILENAME, REACTANT_XYZ_FILENAME
    start_xyz_text = (ctx.rxn_dir / start_filename).read_text(encoding="utf-8")
    end_xyz_text = (ctx.rxn_dir / end_filename).read_text(encoding="utf-8")
    run_dir = ctx.rxn_dir / run_dirname
    run_dir.mkdir(parents=True, exist_ok=True)
    _reset_acp_attempt_state(run_dir)
    xtb_settings = _g2_settings(ctx.config, "xtb")
    timeout_raw = xtb_settings.get("timeout_seconds", _DEFAULT_XTB_TIMEOUT_SECONDS)
    outcome = run_xtb_path_acp_attempt(
        reaction_dir=ctx.rxn_dir,
        config=ctx.config,
        direction=direction,
        execution_id=f"pes2ts-g2-{uuid4().hex}",
        attempt_id=f"attempt-{uuid4().hex}",
        timeout_seconds=float(timeout_raw),
        charge=ctx.charge,
        uhf=ctx.uhf,
        start_xyz_text=start_xyz_text,
        end_xyz_text=end_xyz_text,
        **_acp_wiring(ctx.config),
    )
    return outcome.frames, outcome.failure_detail, outcome.attempt


def _normalize(frames: tuple[Frame, ...]) -> tuple[tuple[Frame, ...], list[float]]:
    """Reverse P→R frames to R→P, recalibrating energies against new frame 0."""
    ordered = tuple(reversed(frames))
    base = ordered[0].energy
    raw = [float(frame.energy) for frame in ordered]
    normalized = tuple(
        Frame(frame.elements, frame.coordinates, None if frame.energy is None else frame.energy - base)
        for frame in ordered
    )
    return normalized, raw


def _metric_rows(
    ctx: _ReactionContext, frames: tuple[Frame, ...], raw: Sequence[float] | None
) -> list[Metric]:
    rows = frame_metrics(
        frames,
        reaction_id=ctx.reaction_id,
        reactant_coords=ctx.reactant_coords,
        product_coords=ctx.product_coords,
        elements=ctx.elements,
        r_pairs=ctx.r_pairs,
        p_pairs=ctx.p_pairs,
        events=ctx.events,
        config=ctx.config,
    )
    for index, row in enumerate(rows):
        row["energy_rel_kcal_raw"] = float(row["energy_rel_kcal"]) if raw is None else raw[index]
    return rows


def _sources(ctx: _ReactionContext, run_dir: Path) -> dict[str, str]:
    sources = {
        "g1_document": sha256_file(ctx.g1_path),
        REACTANT_XYZ_FILENAME: sha256_file(ctx.rxn_dir / REACTANT_XYZ_FILENAME),
        PRODUCT_XYZ_FILENAME: sha256_file(ctx.rxn_dir / PRODUCT_XYZ_FILENAME),
    }
    for relative in _ACP_SOURCE_KEYS:
        candidate = run_dir / relative
        if candidate.is_file():
            sources[relative] = sha256_file(candidate)
    return sources


def _code_str(code: RejectionCode | None) -> str | None:
    return None if code is None else code.value


def _terminal_run_dir(ctx: _ReactionContext, direction: str) -> Path:
    return ctx.rxn_dir / (RUN_DIRNAME if direction == DIRECTION_FORWARD else REVERSE_RUN_DIRNAME)


def _path_inp_text(config: Mapping[str, Any]) -> str:
    """The frozen ``$path`` text the ACP request carries, from ``g2.path``."""
    return build_path_inp_text(_g2_settings(config, "path"))


def _reaction_document(
    ctx: _ReactionContext,
    *,
    verdict: Verdict,
    direction: str,
    attempts: list[dict[str, Any]],
    rows: list[Metric] | None,
    digest: str,
) -> dict[str, Any]:
    """One ``g2_path_v1`` terminal document (digests and scalars only)."""
    recovered = direction == DIRECTION_REVERSE and verdict.status == STATUS_VALID
    record = ctx.endpoint_record
    endpoints = {
        key: record[key]
        for key in ("multiplicity_basis", "metrics", "global_frame_basis")
        if key in record
    }
    endpoints["n_candidates"] = len(record.get("candidates") or ())
    return reaction_document(
        reaction_id=ctx.reaction_id,
        status=str(verdict.status),
        failure_code=_code_str(verdict.failure_code),
        failure_detail=verdict.detail,
        direction=direction,
        direction_recovered=recovered,
        sources=_sources(ctx, _terminal_run_dir(ctx, direction)),
        endpoints=endpoints,
        attempts=attempts,
        frames=rows or (),
        validity=verdict.summary,
        config_digest=digest,
    )


def _execute_reaction(
    ctx: _ReactionContext, *, enabled: bool, triggers: frozenset[RejectionCode]
) -> tuple[dict[str, Any], LedgerFailure | None]:
    """Run forward, retry reversed when triggered, and write the terminal artifacts."""
    forward_frames, forward_failure, forward_attempt = _run_attempt(ctx, RUN_DIRNAME, DIRECTION_FORWARD)
    verdict = _judge(forward_frames, forward_failure, ctx)
    forward_attempt["failure_code"] = _code_str(verdict.failure_code)
    attempts = [forward_attempt]
    direction, terminal_frames, terminal_raw = DIRECTION_FORWARD, forward_frames, None
    if verdict.status != STATUS_VALID and enabled and verdict.failure_code in triggers:
        reverse_frames, reverse_failure, reverse_attempt = _run_attempt(
            ctx, REVERSE_RUN_DIRNAME, DIRECTION_REVERSE
        )
        attempts.append(reverse_attempt)
        direction = DIRECTION_REVERSE
        if reverse_frames is None:
            verdict = _judge(None, reverse_failure, ctx)
        else:
            normalized, terminal_raw = _normalize(reverse_frames)
            verdict = _judge(normalized, None, ctx)
            terminal_frames = normalized
        reverse_attempt["failure_code"] = _code_str(verdict.failure_code)
    rows = None if terminal_frames is None else _metric_rows(ctx, terminal_frames, terminal_raw)
    digest = config_digest(
        ctx.config,
        xtb_sha256=str(attempts[-1].get("request_sha256")),
        path_inp=_path_inp_text(ctx.config),
    )
    document = _reaction_document(
        ctx,
        verdict=verdict,
        direction=direction,
        attempts=attempts,
        rows=rows,
        digest=digest,
    )
    write_json(ctx.rxn_dir / DOCUMENT_FILENAME, document)
    if rows is not None:
        write_frames_parquet(ctx.rxn_dir / FRAMES_FILENAME, rows)
    if verdict.status == STATUS_VALID:
        return document, None
    detail = verdict.detail or f"terminal status {verdict.status}"
    return document, (
        ctx.reaction_id, _code_str(verdict.failure_code), detail,
        str(ctx.rxn_dir / DOCUMENT_FILENAME),
    )


def _xtb_fingerprint(
    config: Mapping[str, Any], last_attempt: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Provenance block for the manifest ``xtb`` field (ACP-executed runs).

    ``request_sha256``/``manifest_sha256`` come from the last attempt of the
    last attempted reaction in sorted selected order — freshly run or resumed
    from the persisted terminal document, so an idempotent re-run reproduces
    the fresh run's block (all ``None`` only when the batch holds no attempt
    at all).  ``path_inp`` is the frozen ``$path`` text the ACP request
    carries, regenerated from ``g2.path``.  The executable digest, argv,
    version line, OMP/seed fields are not surfaced by the ACP CLI transport
    and stay ``None`` (never fabricated).  ``config_digest`` keeps consuming
    only ``sha256``/``path_inp``.
    """
    fingerprint: dict[str, Any] = {
        "sha256": None,
        "request_sha256": None,
        "manifest_sha256": None,
        "path_inp": _path_inp_text(config),
        "version": None,
        "argv": None,
        "omp_num_threads": None,
        "seed_supported": None,
        "seed": None,
    }
    if last_attempt is not None:
        fingerprint["request_sha256"] = last_attempt.get("request_sha256")
        fingerprint["manifest_sha256"] = last_attempt.get("manifest_sha256")
        fingerprint["version"] = last_attempt.get("xtb_version_line")
    return fingerprint


def _write_batch_artifacts(
    config: Mapping[str, Any],
    *,
    selected: Sequence[str],
    documents: Mapping[str, Mapping[str, Any]],
    run_counts: dict[str, int],
) -> None:
    interim_dir, manifests_dir = _paths(config)
    shard_size = _shard_size(config)
    summary_documents = []
    for reaction_id in selected:
        document = documents.get(reaction_id) or _terminal_document(
            _rxn_dir(interim_dir, reaction_id, shard_size) / DOCUMENT_FILENAME
        )
        if document is not None:
            summary_documents.append(document)
    summary_rows = build_summary(summary_documents)
    write_summary(interim_dir / SUMMARY_FILENAME, summary_rows)
    last_attempt: Mapping[str, Any] | None = None
    for reaction_id in selected:
        document = documents.get(reaction_id) or _terminal_document(
            _rxn_dir(interim_dir, reaction_id, shard_size) / DOCUMENT_FILENAME
        )
        attempts = (document or {}).get("attempts")
        if attempts:
            last_attempt = attempts[-1]
    fingerprint = _xtb_fingerprint(config, last_attempt)
    write_manifest(
        manifests_dir / MANIFEST_FILENAME, summary_rows,
        run_counts=run_counts, xtb_fingerprint=fingerprint, config=config,
    )
    categories_by_reaction: dict[str, Mapping[str, Any]] = {}
    for reaction_id in selected:
        doc_path = reaction_change_path(interim_dir, reaction_id, shard_size)
        if doc_path.is_file():
            g1_document = read_json(doc_path)
            if isinstance(g1_document, Mapping) and isinstance(g1_document.get("categories"), Mapping):
                categories_by_reaction[reaction_id] = g1_document["categories"]
    current_by_code = Counter(
        str(row["failure_code"]) for row in summary_rows if row["failure_code"]
    )
    historical_failed_by_code = Counter(
        str(record.get("code"))
        for record in _ledger_records(manifests_dir)
        if record.get("stage") == STAGE and record.get("code")
    )
    write_coverage(
        manifests_dir / COVERAGE_FILENAME, summary_rows,
        categories_by_reaction=categories_by_reaction,
        current_by_code=dict(current_by_code),
        historical_failed_by_code=dict(historical_failed_by_code),
    )


def run_ids(ids: Sequence[str], *, config: Mapping[str, Any], force: bool = False) -> RunReport:
    """Run the xTB PATH stage for every selected reaction and close the batch.

    Ids with a terminal ``reaction_path.json`` are skipped unless *force*
    (idempotency: no new ACP calls, unchanged terminal artifact digests).  An
    id without prepared endpoints raises :class:`InfrastructureError` naming
    the ``g2 prepare`` hint -- never a silent skip.  Terminal failures append
    one idempotent ledger record; the batch always ends with summary, manifest,
    and coverage reflecting the latest batch (ledger history is append-only).
    """
    interim_dir, manifests_dir = _paths(config)
    shard_size = _shard_size(config)
    eligible = _load_eligible(config)
    rows_by_id = _load_inventory(config)
    retry_settings = _g2_settings(config, "reverse_retry")
    enabled = bool(retry_settings.get("enabled", True))
    triggers = frozenset(
        RejectionCode(str(code))
        for code in retry_settings.get("trigger_codes", DEFAULT_REVERSE_RETRY_TRIGGERS)
    )
    selected = sorted(dict.fromkeys(str(reaction_id) for reaction_id in ids))
    documents: dict[str, Mapping[str, Any]] = {}
    skipped = 0
    failures: list[LedgerFailure] = []
    for reaction_id in selected:
        rxn_dir = _rxn_dir(interim_dir, reaction_id, shard_size)
        if not force and _terminal_document(rxn_dir / DOCUMENT_FILENAME) is not None:
            skipped += 1
            continue
        if reaction_id not in eligible:
            failures.append((
                reaction_id, RejectionCode.G2_NOT_ELIGIBLE.value,
                "reaction is not in the eligible list", ELIGIBLE_FILENAME,
            ))
            continue
        g1, failure = _g1_check(interim_dir, reaction_id, shard_size)
        if failure is not None:
            failures.append(failure)
            continue
        document, g1_path = g1
        row = rows_by_id.get(reaction_id)
        if row is None:
            raise InfrastructureError(f"missing inventory row for {reaction_id}; {INVENTORY_HINT}")
        try:
            ctx = _reaction_context(reaction_id, rxn_dir, document, row, config, g1_path)
        except (XtbOutputError, ValueError) as error:
            raise InfrastructureError(
                f"{error}; run `g2 prepare` first for {reaction_id}"
            ) from None
        terminal, failure = _execute_reaction(ctx, enabled=enabled, triggers=triggers)
        documents[reaction_id] = terminal
        if failure is not None:
            failures.append(failure)
    _append_ledger(manifests_dir, failures)
    run_counts = {
        "n_selected": len(selected), "n_attempted": len(documents), "n_skipped": skipped,
    }
    _write_batch_artifacts(config, selected=selected, documents=documents, run_counts=run_counts)
    logger.info(
        "G2 run: %d selected, %d attempted, %d skipped, %d failed",
        len(selected), len(documents), skipped, len(failures),
    )
    return RunReport(**run_counts)


__all__ = [
    "COVERAGE_FILENAME",
    "MANIFEST_FILENAME",
    "PREPARE_HINT",
    "SUMMARY_FILENAME",
    "InfrastructureError",
    "PrepareReport",
    "RunReport",
    "prepare_ids",
    "run_ids",
    "select_reaction_ids",
]
