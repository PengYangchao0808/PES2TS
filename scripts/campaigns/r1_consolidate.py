#!/usr/bin/env python3
"""R1 experiment-matrix + run-manifest consolidation (V5_1A_R1_REPAIR).

Regenerates, from disk truth (READ-ONLY over run dirs — nothing under
``outputs/pesn/v5_1*/runs/`` is ever written):

* ``outputs/pesn/v5_1_r1/experiment_matrix.csv`` — one row per
  ``run_record.json`` found under ``outputs/pesn/v5_1/runs`` (the 21
  V5.1a runs) and ``outputs/pesn/v5_1_r1/runs`` (A0 baseline fill +
  R1 dev/confirm arms as they land).  Columns (frozen order)::

      run_id, campaign, stage, arm, fold, seed, updates_budget, status,
      best_update, inner_rmse_over_zero, inner_sre_median,
      inner_near_ts_cos, inner_sign_accuracy, alpha_median,
      dead_zone_fraction, source, reused_from, config_sha256,
      data_manifest_sha256, loss_schema_version, wall_s, notes

  ``source`` ∈ {``v5_1a_reuse``, ``r1_baseline_fill``, ``r1_dev``,
  ``r1_confirm``}; stage: V5.1a runs → ``dev_reference`` (folds 0–2)
  / ``confirm_reference`` (folds 3–4), R1 runs → the stage recorded in
  ``sprint/r1_decisions.jsonl`` (fallback: B0 → ``baseline_fill``,
  folds 0–2 → ``dev``, folds 3–4 → ``confirm``); ``wall_s`` from the
  campaign decision logs (legacy ``v5_1/sprint/stage_decisions.jsonl``
  subprocess rows keyed by ``--out``, R1 ``r1_decisions.jsonl`` launch
  rows keyed by ``run_id``).

* ``outputs/pesn/v5_1_r1/run_manifest.json`` — every run with relative
  paths + sha256 of run_record/metrics/checkpoint_best (schema
  ``pesn_v5_1_r1_run_manifest_v1``).

Development bookkeeping only (inner-val reported numbers are copied,
never recomputed).  Deterministic and re-runnable: safe to run now and
again at campaign end.

Usage::

    /root/miniforge3/bin/python3 scripts/campaigns/r1_consolidate.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MANIFEST_SCHEMA: Final[str] = "pesn_v5_1_r1_run_manifest_v1"
CAMPAIGN_ID: Final[str] = "V5_1A_R1_REPAIR"

LEGACY_RUNS_ROOT: Final[Path] = PROJECT_ROOT / "outputs/pesn/v5_1/runs"
R1_RUNS_ROOT: Final[Path] = PROJECT_ROOT / "outputs/pesn/v5_1_r1/runs"
LEGACY_DECISIONS: Final[Path] = (
    PROJECT_ROOT / "outputs/pesn/v5_1/sprint/stage_decisions.jsonl")
R1_DECISIONS: Final[Path] = (
    PROJECT_ROOT / "outputs/pesn/v5_1_r1/sprint/r1_decisions.jsonl")
OUT_MATRIX: Final[Path] = (
    PROJECT_ROOT / "outputs/pesn/v5_1_r1/experiment_matrix.csv")
OUT_MANIFEST: Final[Path] = (
    PROJECT_ROOT / "outputs/pesn/v5_1_r1/run_manifest.json")

MATRIX_FIELDS: Final[tuple[str, ...]] = (
    "run_id", "campaign", "stage", "arm", "fold", "seed",
    "updates_budget", "status", "best_update", "inner_rmse_over_zero",
    "inner_sre_median", "inner_near_ts_cos", "inner_sign_accuracy",
    "alpha_median", "dead_zone_fraction", "source", "reused_from",
    "config_sha256", "data_manifest_sha256", "loss_schema_version",
    "wall_s", "notes",
)

SOURCE_LEGACY: Final[str] = "v5_1a_reuse"
SOURCE_FILL: Final[str] = "r1_baseline_fill"
SOURCE_DEV: Final[str] = "r1_dev"
SOURCE_CONFIRM: Final[str] = "r1_confirm"

DEV_FOLDS: Final[frozenset[int]] = frozenset({0, 1, 2})
CONFIRM_FOLDS: Final[frozenset[int]] = frozenset({3, 4})


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> dict | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return doc if isinstance(doc, dict) else None


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a torn tail line from a concurrent writer
        if isinstance(doc, dict):
            rows.append(doc)
    return rows


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    except OSError:
        return None
    return h.hexdigest()


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _csv(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return str(value)


# ---------------------------------------------------------------------------
# wall-time logs (pure readers — unit-testable)
# ---------------------------------------------------------------------------

def legacy_wall_times(rows: Sequence[Mapping]) -> dict[str, float]:
    """run_id → wall_s from legacy ``stage_decisions.jsonl`` subprocess
    rows (keyed by the ``--out`` argument's basename)."""
    out: dict[str, float] = {}
    for row in rows:
        if row.get("decision") != "subprocess":
            continue
        argv = list(row.get("argv") or [])
        if "--out" not in argv or "wall_s" not in row:
            continue
        run_id = Path(str(argv[argv.index("--out") + 1])).name
        try:
            out[run_id] = float(row["wall_s"])
        except (TypeError, ValueError):
            continue
    return out


def r1_stage_and_wall(rows: Sequence[Mapping]) -> dict[str, dict]:
    """run_id → {stage, wall_s} from R1 ``r1_decisions.jsonl``.

    Later rows win (the log is append-only across resumes); only rows
    carrying a ``run_id`` and a known stage are considered, ``wall_s``
    is taken from ``launch`` rows (the only ones that time a run).
    """
    stages = ("baseline_fill", "dev", "confirm", "screen")
    out: dict[str, dict] = {}
    for row in rows:
        run_id = row.get("run_id")
        if not run_id:
            continue
        stage = row.get("stage")
        entry = out.setdefault(str(run_id), {})
        if stage in stages:
            entry["stage"] = str(stage)
        if row.get("decision") == "launch" and "wall_s" in row:
            try:
                entry["wall_s"] = float(row["wall_s"])
            except (TypeError, ValueError):
                pass
    return out


# ---------------------------------------------------------------------------
# row building (pure — unit-testable)
# ---------------------------------------------------------------------------

def classify_legacy(fold: int) -> tuple[str, str]:
    """V5.1a run → (stage, notes). Folds 0–2 are dev references,
    folds 3–4 confirm references (protocol partitions)."""
    if int(fold) in DEV_FOLDS:
        return "dev_reference", ""
    if int(fold) in CONFIRM_FOLDS:
        return "confirm_reference", ""
    return "", f"unexpected fold {fold}"


def classify_r1(arm: str, fold: int, stage_hint: str | None
                ) -> tuple[str, str]:
    """R1 run → (stage, source). Stage prefers the decision log."""
    if stage_hint in ("baseline_fill", "dev", "confirm"):
        stage = stage_hint
    elif str(arm) == "B0":
        stage = "baseline_fill"
    elif int(fold) in CONFIRM_FOLDS:
        stage = "confirm"
    else:
        stage = "dev"
    if stage == "baseline_fill":
        source = SOURCE_FILL
    elif stage == "confirm":
        source = SOURCE_CONFIRM
    else:
        source = SOURCE_DEV
    return stage, source


def build_row(*, run_record: Mapping, run_dir: Path, source: str,
              stage: str, campaign: str, wall_s: float | None,
              status_override: str | None = None) -> dict:
    """One experiment-matrix row from a run_record (+ status/metrics
    fallbacks). Missing fields stay empty — never fabricated."""
    arm_block = dict(run_record.get("arm") or {})
    inner = dict(run_record.get("inner_val_score") or {})
    status = status_override
    if status is None:
        status_doc = _read_json(run_dir / "status.json")
        status = str(
            (status_doc or {}).get("status")
            or run_record.get("status") or "")
    gv = dict(run_record.get("gv") or {})
    return {
        "run_id": str(run_record.get("run_id") or run_dir.name),
        "campaign": campaign,
        "stage": stage,
        "arm": str(arm_block.get("arm_id", "")),
        "fold": run_record.get("fold"),
        "seed": run_record.get("seed"),
        "updates_budget": run_record.get("updates_budget"),
        "status": status,
        "best_update": run_record.get("best_update"),
        "inner_rmse_over_zero": inner.get("rmse_over_zero"),
        "inner_sre_median": inner.get("sre_median"),
        "inner_near_ts_cos": inner.get("near_ts_cos"),
        "inner_sign_accuracy": run_record.get("sign_accuracy"),
        "alpha_median": run_record.get("alpha_median"),
        "dead_zone_fraction": run_record.get("dead_zone_fraction"),
        "source": source,
        "reused_from": (_rel(run_dir) if source == SOURCE_LEGACY else ""),
        "config_sha256": gv.get("config_sha256"),
        "data_manifest_sha256": gv.get("data_manifest_sha256"),
        "loss_schema_version": run_record.get("loss_schema_version"),
        "wall_s": wall_s,
        "notes": "",
    }


def _row_note(row: Mapping, run_record: Mapping, run_dir: Path) -> str:
    notes: list[str] = []
    run_id = str(row["run_id"])
    if row["source"] == SOURCE_LEGACY:
        if run_id.endswith("_d1"):
            notes.append("v5_1a diagnose-stage run")
        elif str((run_record.get("arm") or {}).get("arm_id")) == "B0":
            notes.append("v5_1a A0 reuse cell (reused as A0)")
        else:
            notes.append("v5_1a reference arm (not in A0 matrix)")
    elif row["source"] == SOURCE_FILL:
        notes.append("A0 baseline fill with original v5_1_sprint.yaml")
    elif row["source"] == SOURCE_DEV:
        notes.append("R1 dev arm")
    else:
        notes.append("R1 confirm arm")
    if str(row["status"]) not in ("complete", "censored_patience",
                                  "censored_screen"):
        notes.append(f"status={row['status'] or 'unknown'}")
    if row["inner_rmse_over_zero"] is None:
        notes.append("inner metrics pending")
    return "; ".join(notes)


def collect_rows(*, legacy_root: Path, r1_root: Path,
                 legacy_decisions: Path,
                 r1_decisions: Path) -> tuple[list[dict], list[dict]]:
    """Rows for the matrix + per-run manifest entries (disk truth)."""
    legacy_wall = legacy_wall_times(_read_jsonl(legacy_decisions))
    r1_meta = r1_stage_and_wall(_read_jsonl(r1_decisions))

    rows: list[dict] = []
    entries: list[dict] = []

    def _files(run_dir: Path) -> dict:
        files: dict[str, dict] = {}
        for name in ("run_record.json", "metrics.json", "status.json",
                     "checkpoints/checkpoint_best.pt", "console.log",
                     "eval.jsonl"):
            path = run_dir / name
            if path.is_file():
                files[name] = {
                    "path": _rel(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
        return files

    # -- V5.1a legacy runs (the 21 v5_1a ones) ----------------------------
    for run_dir in sorted(p for p in legacy_root.glob("*/")
                          if (p / "run_record.json").is_file()):
        run_record = _read_json(run_dir / "run_record.json")
        if run_record is None:
            rows.append({
                "run_id": run_dir.name, "campaign": "",
                "stage": "", "arm": "", "fold": "", "seed": "",
                "updates_budget": "", "status": "unreadable_run_record",
                "best_update": "", "inner_rmse_over_zero": "",
                "inner_sre_median": "", "inner_near_ts_cos": "",
                "inner_sign_accuracy": "", "alpha_median": "",
                "dead_zone_fraction": "", "source": SOURCE_LEGACY,
                "reused_from": _rel(run_dir), "config_sha256": "",
                "data_manifest_sha256": "", "loss_schema_version": "",
                "wall_s": "", "notes": "unreadable run_record.json",
            })
            continue
        fold = int(run_record.get("fold", -1))
        stage, stage_note = classify_legacy(fold)
        campaign = str(run_record.get("campaign_id") or "V5_1A_SPRINT")
        wall = legacy_wall.get(str(run_record.get("run_id") or run_dir.name))
        row = build_row(
            run_record=run_record, run_dir=run_dir, source=SOURCE_LEGACY,
            stage=stage, campaign=campaign, wall_s=wall)
        note = _row_note(row, run_record, run_dir)
        if stage_note:
            note = f"{note}; {stage_note}".strip("; ")
        row["notes"] = note
        rows.append(row)
        gv = dict(run_record.get("gv") or {})
        entries.append({
            "run_id": row["run_id"], "source": SOURCE_LEGACY,
            "stage": stage, "campaign": campaign, "arm": row["arm"],
            "fold": row["fold"], "seed": row["seed"],
            "status": row["status"],
            "run_dir": _rel(run_dir),
            "config_path": _config_path(run_record),
            "config_sha256": gv.get("config_sha256"),
            "data_manifest_sha256": gv.get("data_manifest_sha256"),
            "loss_schema_version": run_record.get("loss_schema_version"),
            "inner_val_split_hash": run_record.get("inner_val_split_hash"),
            "wall_s": wall,
            "files": _files(run_dir),
        })

    # -- R1 runs (baseline fill + dev/confirm arms) -----------------------
    for run_dir in sorted(p for p in r1_root.glob("*/")
                          if (p / "run_record.json").is_file()):
        run_record = _read_json(run_dir / "run_record.json")
        run_id = str((run_record or {}).get("run_id") or run_dir.name)
        meta = r1_meta.get(run_id, {})
        if run_record is None:
            rows.append({
                "run_id": run_id, "campaign": CAMPAIGN_ID,
                "stage": meta.get("stage", ""), "arm": "", "fold": "",
                "seed": "", "updates_budget": "",
                "status": "unreadable_run_record", "best_update": "",
                "inner_rmse_over_zero": "", "inner_sre_median": "",
                "inner_near_ts_cos": "", "inner_sign_accuracy": "",
                "alpha_median": "", "dead_zone_fraction": "",
                "source": SOURCE_FILL, "reused_from": "",
                "config_sha256": "", "data_manifest_sha256": "",
                "loss_schema_version": "", "wall_s": meta.get("wall_s"),
                "notes": "unreadable run_record.json",
            })
            continue
        fold = int(run_record.get("fold", -1))
        arm = str((run_record.get("arm") or {}).get("arm_id", "B0"))
        stage, source = classify_r1(arm, fold, meta.get("stage"))
        campaign = str(run_record.get("campaign_id") or CAMPAIGN_ID)
        row = build_row(
            run_record=run_record, run_dir=run_dir, source=source,
            stage=stage, campaign=campaign, wall_s=meta.get("wall_s"))
        row["notes"] = _row_note(row, run_record, run_dir)
        rows.append(row)
        gv = dict(run_record.get("gv") or {})
        entries.append({
            "run_id": run_id, "source": source, "stage": stage,
            "campaign": campaign, "arm": row["arm"],
            "fold": row["fold"], "seed": row["seed"],
            "status": row["status"],
            "run_dir": _rel(run_dir),
            "config_path": _config_path(run_record),
            "config_sha256": gv.get("config_sha256"),
            "data_manifest_sha256": gv.get("data_manifest_sha256"),
            "loss_schema_version": run_record.get("loss_schema_version"),
            "inner_val_split_hash": run_record.get("inner_val_split_hash"),
            "wall_s": meta.get("wall_s"),
            "files": _files(run_dir),
        })

    source_order = {SOURCE_LEGACY: 0, SOURCE_FILL: 1,
                    SOURCE_DEV: 2, SOURCE_CONFIRM: 3}

    def _sort_key(row: Mapping):
        fold = row["fold"]
        seed = row["seed"]
        return (source_order.get(str(row["source"]), 9),
                int(fold) if isinstance(fold, int) else 99,
                int(seed) if isinstance(seed, int) else 99,
                str(row["run_id"]))

    rows.sort(key=_sort_key)
    entries.sort(key=lambda e: (
        source_order.get(str(e["source"]), 9),
        int(e["fold"]) if isinstance(e["fold"], int) else 99,
        int(e["seed"]) if isinstance(e["seed"], int) else 99,
        str(e["run_id"])))
    return rows, entries


def _config_path(run_record: Mapping) -> str | None:
    raw = run_record.get("config_path")
    if not raw:
        return None
    path = Path(str(raw))
    return _rel(path) if path.is_absolute() else str(path)


# ---------------------------------------------------------------------------
# writers (atomic; never inside runs/)
# ---------------------------------------------------------------------------

def write_matrix(rows: Sequence[Mapping], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(MATRIX_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv(row.get(k)) for k in MATRIX_FIELDS})
    tmp.replace(out_path)


def write_manifest(entries: Sequence[Mapping], out_path: Path,
                   *, legacy_root: Path, r1_root: Path) -> None:
    by_source: dict[str, int] = {}
    for entry in entries:
        by_source[str(entry["source"])] = (
            by_source.get(str(entry["source"]), 0) + 1)
    doc = {
        "schema_version": MANIFEST_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "development_bookkeeping",
        "disclaimer": (
            "READ-ONLY consolidation over run dirs: paths + sha256 of "
            "landed artifacts; no run output is modified."),
        "roots": {
            "legacy_runs": _rel(legacy_root),
            "r1_runs": _rel(r1_root),
        },
        "counts": {"total": len(entries), "by_source": dict(sorted(
            by_source.items()))},
        "runs": list(entries),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    tmp.write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8")
    json.loads(tmp.read_text(encoding="utf-8"))  # validity re-check
    tmp.replace(out_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="r1_consolidate.py",
        description="Regenerate the R1 experiment matrix CSV + run "
                    "manifest from landed run_record.json files "
                    "(READ-ONLY over run dirs)")
    p.add_argument("--legacy-runs-root", default=str(LEGACY_RUNS_ROOT))
    p.add_argument("--r1-runs-root", default=str(R1_RUNS_ROOT))
    p.add_argument("--legacy-decisions", default=str(LEGACY_DECISIONS))
    p.add_argument("--r1-decisions", default=str(R1_DECISIONS))
    p.add_argument("--out-matrix", default=str(OUT_MATRIX))
    p.add_argument("--out-manifest", default=str(OUT_MANIFEST))
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    legacy_root = Path(args.legacy_runs_root)
    r1_root = Path(args.r1_runs_root)
    if not legacy_root.is_dir():
        raise SystemExit(f"legacy runs root not found: {legacy_root}")

    rows, entries = collect_rows(
        legacy_root=legacy_root, r1_root=r1_root,
        legacy_decisions=Path(args.legacy_decisions),
        r1_decisions=Path(args.r1_decisions))

    matrix_path = Path(args.out_matrix)
    manifest_path = Path(args.out_manifest)
    write_matrix(rows, matrix_path)
    write_manifest(entries, manifest_path,
                   legacy_root=legacy_root, r1_root=r1_root)

    n_legacy = sum(1 for r in rows if r["source"] == SOURCE_LEGACY)
    n_r1 = len(rows) - n_legacy
    print(f"[r1_consolidate] {len(rows)} runs "
          f"({n_legacy} v5_1a_reuse, {n_r1} r1)")
    if n_legacy != 21:
        print(f"[r1_consolidate] WARNING: expected 21 v5_1a runs, "
              f"found {n_legacy}", file=sys.stderr)
    print(f"[r1_consolidate] wrote {matrix_path}")
    print(f"[r1_consolidate] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
