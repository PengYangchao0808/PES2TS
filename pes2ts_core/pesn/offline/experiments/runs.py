"""Experiment run records: identity, events, resources, outputs (B4).

Run identity ``g3p1-YYYYMMDD-<config_hash8>-seed<seed>-r<attempt>`` — every
new attempt gets a new id; a resume keeps the id and appends events.  The
record binds config/dataset/split/feature/label hashes, model parameter
count, per-epoch history, checkpoint outputs, resource measurements, and
every failure.  All writes append; nothing is overwritten.
"""

from __future__ import annotations

import hashlib
import json
import resource
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Mapping

from pes2ts_core.pesn.contracts.artifacts import RUN_SCHEMA_VERSION, validate_run_record
from pes2ts_core.utils.hashing import JSONValue, stable_json_dumps


def sha256_text(text: str) -> str:
    """Hex sha256 of a UTF-8 string (local; utils.hashing stays untouched)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

RUN_SCHEMA: Final[str] = RUN_SCHEMA_VERSION
RUNS_DIR_NAME: Final[str] = "outputs/pesn/g3_p1"


class RunRecordError(RuntimeError):
    """Typed run-record failure."""


def make_run_id(*, config_sha256: str, seed: int, attempt: int) -> str:
    date = datetime.now(UTC).strftime("%Y%m%d")
    return f"g3p1-{date}-{config_sha256[:8]}-seed{seed}-r{attempt}"


def config_content_sha256(config: Mapping[str, JSONValue]) -> str:
    return sha256_text(stable_json_dumps(dict(config)))


class RunRecord:
    """Append-only run journal under ``outputs/pesn/g3_p1/<run_id>/``."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_path = self.run_dir / "run.json"
        self.events_path = self.run_dir / "events.jsonl"
        if not self.run_path.is_file():
            raise RunRecordError(f"run record missing: {self.run_path}")
        self.document: dict[str, JSONValue] = json.loads(self.run_path.read_text())

    @classmethod
    def start(
        cls,
        *,
        runs_root: str | Path,
        run_id: str,
        config: Mapping[str, JSONValue],
        config_sha256: str,
        dataset_manifest_sha256: str,
        split_manifest_sha256: str,
        feature_spec_version: str,
        label_spec_sha256: str,
        seed: int,
        n_parameters: int,
        backend_versions: Mapping[str, str],
        training_mode: str | None = None,
        called_config_sha256: str | None = None,
        gv: Mapping[str, object] | None = None,
    ) -> "RunRecord":
        run_dir = Path(runs_root) / run_id
        run_path = run_dir / "run.json"
        if run_path.exists():
            raise RunRecordError(f"run already exists (resume instead): {run_path}")
        run_dir.mkdir(parents=True, exist_ok=True)
        document: dict[str, JSONValue] = {
            "schema_version": RUN_SCHEMA,
            "run_id": run_id,
            "status": "running",
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "seed": seed,
            "config": dict(config),
            "config_sha256": config_sha256,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "split_manifest_sha256": split_manifest_sha256,
            "feature_spec_version": feature_spec_version,
            "label_spec_sha256": label_spec_sha256,
            "n_parameters": n_parameters,
            "backend_versions": dict(backend_versions),
            "epochs": [],
            "events": [],
            "outputs": {},
            "resources": {},
            "failures": [],
        }
        if training_mode is not None:
            document["training_mode"] = training_mode
        if gv is not None:
            # G–V provenance: campaign id, primary/supporting G stages, gates,
            # map version and code commit.  New V5+ runs must pass this block;
            # legacy V4 records may omit it.
            from pes2ts_core.gv_registry import load_gv_map, validate_gv_run_record

            document["gv"] = dict(gv)
            validate_gv_run_record(document, load_gv_map())
        if called_config_sha256 is not None and called_config_sha256 != config_sha256:
            document["config_provenance"] = {
                "called_config_sha256": called_config_sha256,
                "effective_config_sha256": config_sha256,
                "note": (
                    "the effective config is the called config with the "
                    "resolved training mode written back; both digests are "
                    "kept so the override is auditable"
                ),
            }
        validate_run_record(document)
        run_path.write_text(json.dumps(document, indent=2))
        return cls(run_dir)

    def append_event(self, kind: str, **fields: JSONValue) -> None:
        entry: dict[str, JSONValue] = {
            "ts": datetime.now(UTC).isoformat(timespec="seconds"),
            "kind": kind,
            **fields,
        }
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")
        self.document.setdefault("events", []).append(entry)

    def add_epoch(self, record: Mapping[str, JSONValue]) -> None:
        self.document.setdefault("epochs", []).append(dict(record))
        self._flush()

    def update_status(self, status: str, **extra: JSONValue) -> None:
        self.document["status"] = status
        self.document.update(extra)
        self._flush()

    def record_resources(self, **values: JSONValue) -> None:
        self.document.setdefault("resources", {}).update(values)
        peak = peak_rss_mb()
        if peak is None:
            self.document["resources"]["peak_rss_mb"] = None
            self.document["resources"]["peak_rss_mb_unavailable_reason"] = (
                "resource.getrusage unavailable on this platform"
            )
        else:
            self.document["resources"]["peak_rss_mb"] = peak
        self._flush()

    def record_failure(self, stage: str, detail: str) -> None:
        self.document.setdefault("failures", []).append(
            {"stage": stage, "detail": detail}
        )
        self._flush()

    def record_outputs(self, **paths: JSONValue) -> None:
        self.document.setdefault("outputs", {}).update(paths)
        self._flush()

    def finalize(self, status: str) -> None:
        self.document["status"] = status
        self.document["finished_at"] = datetime.now(UTC).isoformat(timespec="seconds")
        resources = self.document.setdefault("resources", {})
        peak = peak_rss_mb()
        if peak is None:
            resources["peak_rss_mb"] = None
            resources["peak_rss_mb_unavailable_reason"] = (
                "resource.getrusage unavailable on this platform"
            )
        else:
            resources["peak_rss_mb"] = peak
        self._flush()

    def _flush(self) -> None:
        validate_run_record(self.document)
        self.run_path.write_text(json.dumps(self.document, indent=2))


def peak_rss_mb() -> float | None:
    """Peak resident set size of this process (Linux ru_maxrss is KiB).

    Returns ``None`` when the platform cannot report it; callers record the
    null value with an explicit reason instead of guessing.
    """
    try:
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    except (AttributeError, ValueError, OSError):  # pragma: no cover
        return None


__all__ = [
    "RUNS_DIR_NAME",
    "RUN_SCHEMA",
    "RunRecord",
    "RunRecordError",
    "config_content_sha256",
    "make_run_id",
    "peak_rss_mb",
]
