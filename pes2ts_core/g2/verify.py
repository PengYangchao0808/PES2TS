"""G2 artifact verification: re-read the persisted tree and reconcile it.

:func:`verify_g2` independently re-reads the G2 summary, manifest, and every
per-reaction path directory and collects every contract violation as a
human-readable problem string — it never repairs anything and never reads the
unified rejection ledger (historical failure entries never make verify fail).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pes2ts_core.g1.build import reaction_change_path, shard_name
from pes2ts_core.g2 import G2_DIRNAME, G2_PATHS_DIRNAME, SCHEMA_MANIFEST, SCHEMA_PATH
from pes2ts_core.g2.status import G2_STATUSES, STATUS_VALID
from pes2ts_core.utils.hashing import sha256_bytes, sha256_file, stable_json_dumps
from pes2ts_core.utils.jsonio import read_json
from pes2ts_core.utils.parquet_io import read_parquet

#: Keys that must never appear anywhere inside a persisted G2 JSON document.
#: The singular ``coordinate`` of endpoints.json atom records is allowed.
FORBIDDEN_KEYS: tuple[str, ...] = ("coordinates", "EHG", "forces")

#: Filenames every reaction directory must contain.  ``frames.parquet`` is
#: deliberately absent: the pipeline writes no frame file when a failed run
#: produced no parseable frames, so it is required only when the document
#: declares ``frames.n_frames > 0``.
_REACTION_DIR_FILES: tuple[str, ...] = (
    "reaction_path.json", "endpoints.json", "R.xyz", "P.xyz",
)
_DOCUMENT_FILENAME = "reaction_path.json"
_FRAMES_FILENAME = "frames.parquet"
_ENDPOINTS_FILENAME = "endpoints.json"
_SUMMARY_FILENAME = "g2_summary.parquet"
_MANIFEST_FILENAME = "g2_path_manifest.json"
#: ``sources`` keys resolved against the reaction directory itself; every
#: other key is a run-directory filename recorded by the pipeline.
_RXN_DIR_SOURCES = frozenset({"g1_document", "R.xyz", "P.xyz"})
_RUN_DIRNAMES = {"forward": "run", "reverse": "run_reverse"}
_DEFAULT_SHARD_SIZE = 1000


@dataclass(frozen=True, slots=True)
class G2Verification:
    """Outcome of one G2 verification run."""

    n_total: int
    n_valid: int
    n_failed: int
    problems: tuple[str, ...]


def _shard_size(config: Mapping[str, Any]) -> int:
    """The G2 shard width (shared with the G1 document lookup), as in pipeline."""
    g2 = config.get("g2", {})
    return int(g2.get("shard_size", _DEFAULT_SHARD_SIZE)) if isinstance(g2, Mapping) else _DEFAULT_SHARD_SIZE


def _rxn_dir(interim_dir: Path, reaction_id: str, shard_size: int) -> Path:
    return interim_dir / G2_DIRNAME / G2_PATHS_DIRNAME / shard_name(reaction_id, shard_size) / reaction_id


def _forbidden_key_problems(path: Path, value: Any) -> list[str]:
    """Return one problem per forbidden key reachable inside *value* (nested too)."""
    found: set[str] = set()
    stack = [value]
    while stack:
        current = stack.pop()
        if isinstance(current, Mapping):
            for key in FORBIDDEN_KEYS:
                if key in current:
                    found.add(key)
            stack.extend(current.values())
        elif isinstance(current, (list, tuple)):
            stack.extend(current)
    return [f"{path}: forbidden key {key!r} persisted" for key in sorted(found)]


def _source_problems(
    document: Mapping[str, Any], interim_dir: Path, rxn_dir: Path, shard_size: int, path: Path
) -> list[str]:
    """Recompute every digest recorded in the document's sources block."""
    problems: list[str] = []
    sources = document.get("sources")
    if not isinstance(sources, Mapping):
        problems.append(f"{path}: sources block missing")
        return problems
    direction = str(document.get("direction"))
    run_dir = rxn_dir / _RUN_DIRNAMES[direction] if direction in _RUN_DIRNAMES else None
    for key in sorted(sources):
        if key in _RXN_DIR_SOURCES:
            target = (
                reaction_change_path(interim_dir, str(document["reaction_id"]), shard_size)
                if key == "g1_document"
                else rxn_dir / str(key)
            )
        elif run_dir is not None:
            target = run_dir / str(key)
        else:
            problems.append(f"{path}: sources entry {key!r} with unknown direction {direction!r}")
            continue
        if not target.is_file():
            problems.append(f"{path}: sources file {target.name} missing at {target}")
            continue
        if sha256_file(target) != str(sources[key]):
            problems.append(f"{path}: sources digest mismatch for {target.name}")
    return problems


def _summary_agreement_problems(
    row: Mapping[str, Any], document: Mapping[str, Any], path: Path
) -> list[str]:
    """Compare the summary row with the document (status/failure_code/direction/n_frames)."""
    problems: list[str] = []
    frames = document.get("frames")
    n_frames = int(frames.get("n_frames", -1)) if isinstance(frames, Mapping) else -1
    expected = {
        "status": None if document.get("status") is None else str(document.get("status")),
        "failure_code": None if document.get("failure_code") is None else str(document.get("failure_code")),
        "direction": None if document.get("direction") is None else str(document.get("direction")),
        "n_frames": str(n_frames),
    }
    for field in ("status", "failure_code", "direction", "n_frames"):
        row_value = None if row.get(field) is None else str(row[field])
        if row_value != expected[field]:
            problems.append(
                f"{path}: summary {field} {row_value!r} disagrees with document {expected[field]!r}"
            )
    return problems


def _reaction_problems(
    row: Mapping[str, Any], interim_dir: Path, rxn_dir: Path, shard_size: int
) -> list[str]:
    """Return every violation found for one summary row and its directory."""
    reaction_id = str(row["reaction_id"])
    problems: list[str] = []
    for name in _REACTION_DIR_FILES:
        if not (rxn_dir / name).is_file():
            problems.append(f"summary row {reaction_id} has no {rxn_dir / name}")
    document_path = rxn_dir / _DOCUMENT_FILENAME
    if not document_path.is_file():
        return problems
    document = read_json(document_path)
    if not isinstance(document, Mapping):
        problems.append(f"{document_path}: document is not a JSON object")
        return problems
    if document.get("schema_version") != SCHEMA_PATH:
        problems.append(f"{document_path}: schema_version mismatch")
    if str(document.get("reaction_id")) != reaction_id:
        problems.append(f"{document_path}: reaction_id mismatch")
    if str(document.get("status")) not in G2_STATUSES:
        problems.append(f"{document_path}: unknown status {str(document.get('status'))!r}")
    problems.extend(_summary_agreement_problems(row, document, document_path))
    frames_path = rxn_dir / _FRAMES_FILENAME
    frames_summary = document.get("frames")
    n_frames = int(frames_summary.get("n_frames", -1)) if isinstance(frames_summary, Mapping) else -1
    if frames_path.is_file():
        frames = read_parquet(frames_path).to_pylist()
        if len(frames) != n_frames:
            problems.append(
                f"{frames_path}: {len(frames)} rows disagree with document frames.n_frames {n_frames}"
            )
        for foreign in sorted({str(frame.get("reaction_id")) for frame in frames} - {reaction_id}):
            problems.append(f"{frames_path}: frame reaction_id {foreign!r} does not match {reaction_id}")
    elif n_frames > 0:
        problems.append(f"summary row {reaction_id} has no {frames_path}")
    problems.extend(_forbidden_key_problems(document_path, document))
    endpoints_path = rxn_dir / _ENDPOINTS_FILENAME
    if endpoints_path.is_file():
        endpoints = read_json(endpoints_path)
        if isinstance(endpoints, Mapping):
            problems.extend(_forbidden_key_problems(endpoints_path, endpoints))
        else:
            problems.append(f"{endpoints_path}: endpoints record is not a JSON object")
    problems.extend(_source_problems(document, interim_dir, rxn_dir, shard_size, document_path))
    return problems


def _manifest_problems(
    manifest: Mapping[str, Any], rows: list[dict[str, Any]], summary_path: Path, path: Path
) -> list[str]:
    """Reconcile the manifest against the re-read summary rows."""
    problems: list[str] = []
    n_valid = sum(1 for row in rows if str(row.get("status")) == STATUS_VALID)
    derived = {"n_total": len(rows), "n_valid": n_valid, "n_failed": len(rows) - n_valid}
    for name in ("n_total", "n_valid", "n_failed"):
        if manifest.get(name) != derived[name]:
            problems.append(f"manifest {name} disagrees with the summary")
    histogram = Counter(str(row["failure_code"]) for row in rows if row.get("failure_code"))
    expected_by_code = {code: histogram[code] for code in sorted(histogram)}
    if manifest.get("by_code") != expected_by_code:
        problems.append("manifest by_code disagrees with the summary")
    digest = sha256_bytes(stable_json_dumps(rows).encode("utf-8"))
    if manifest.get("summary_sha256") != digest:
        problems.append(f"manifest summary_sha256 does not match {summary_path.name}")
    if manifest.get("schema_version") != SCHEMA_MANIFEST:
        problems.append(f"{path}: manifest schema_version mismatch")
    run = manifest.get("run")
    if not isinstance(run, Mapping):
        problems.append(f"{path}: manifest run block missing")
    else:
        for name in ("n_selected", "n_attempted", "n_skipped"):
            if name not in run:
                problems.append(f"{path}: manifest run block lacks {name}")
    return problems


def verify_g2(*, config: Mapping[str, Any]) -> G2Verification:
    """Re-read the whole G2 tree and reconcile it; collect problems, fix nothing.

    The unified rejection ledger is deliberately never read: historical
    failure entries are append-only audit records and never make verify fail.
    """
    interim_dir = Path(config["paths"]["interim"])
    manifests_dir = Path(config["paths"]["manifests"])
    summary_path = interim_dir / _SUMMARY_FILENAME
    if not summary_path.is_file():
        raise FileNotFoundError(f"Missing G2 summary {summary_path}; run `g2 run` first")
    manifest_path = manifests_dir / _MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing G2 manifest {manifest_path}; run `g2 run` first")
    shard_size = _shard_size(config)
    rows = read_parquet(summary_path).to_pylist()
    problems: list[str] = []
    summary_ids: set[str] = set()
    for row in rows:
        reaction_id = str(row["reaction_id"])
        summary_ids.add(reaction_id)
        problems.extend(
            _reaction_problems(row, interim_dir, _rxn_dir(interim_dir, reaction_id, shard_size), shard_size)
        )
    paths_root = interim_dir / G2_DIRNAME / G2_PATHS_DIRNAME
    if paths_root.is_dir():
        for stray in sorted(paths_root.rglob(_DOCUMENT_FILENAME)):
            document = read_json(stray)
            reaction_id = (
                str(document.get("reaction_id")) if isinstance(document, Mapping) else stray.parent.name
            )
            if reaction_id not in summary_ids:
                problems.append(
                    f"stray G2 document {stray} has no summary row (reaction_id {reaction_id!r})"
                )
    manifest = read_json(manifest_path)
    if isinstance(manifest, Mapping):
        problems.extend(_manifest_problems(manifest, rows, summary_path, manifest_path))
    else:
        problems.append(f"{manifest_path}: manifest is not a JSON object")
    n_valid = sum(1 for row in rows if str(row.get("status")) == STATUS_VALID)
    return G2Verification(len(rows), n_valid, len(rows) - n_valid, tuple(problems))


__all__ = [
    "FORBIDDEN_KEYS",
    "G2Verification",
    "verify_g2",
]
