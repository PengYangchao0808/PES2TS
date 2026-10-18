"""V4.2 legacy (subdivided) label export for uniform trainer consumption.

The V4.2 unbounded-displacement trainer consumes a per-reaction npz carrying
the label-side fields ``action_angstrom`` / ``remaining_arc_length`` /
``is_terminal`` / ``near_zero_action_mask``.  Native data already carries them
(``dual_ts_native_rebuild``).  This module exports the SAME fields for the
LEGACY subdivided ``wp3_features_v41`` shards so both data versions feed one
trainer surface.

Action source (frozen, contract §2 / dual_ts_bc C1): ``action_angstrom`` is the
contract ``label.action_delta`` of the SAME bound teacher table — the
registered-frame displacement ``X_next − X``.  NOTE: the shard ``x_angstrom``
is the frozen centroid-removed X (``X − mean_atoms(X)``, consume-time rule), so
``x[next] − x[curr]`` differs from ``action_delta`` by the per-frame centroid
shift and is NEVER used as the label.  The cross-validation gate
(``|action_angstrom − teacher.action_delta| ≤ 1e-10`` over a 20-state sample)
asserts this alignment explicitly and records the centroid-shift magnitude as
a diagnostic.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

import numpy as np

from pes2ts_core.pesn.contracts.dual_ts import DualTSTeacher
from pes2ts_core.pesn.offline.data.dual_ts_native_rebuild import (
    sha256_file,
)

LEGACY_LABELS_SCHEMA: Final[str] = "pes2ts_dual_ts_legacy_labels_v1"
LEGACY_LABELS_TOOL_VERSION: Final[str] = "pes2ts_dual_ts_legacy_labels_tool_v1"

#: Cross-validation tolerance vs teacher action_delta (contract §2).
ACTION_TOL: Final[float] = 1e-10
#: Number of states to cross-validate (deterministic sample).
CV_SAMPLE_STATES: Final[int] = 20


class LegacyLabelsError(RuntimeError):
    """Typed refusal."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def _require(condition: bool, code: str, detail: str) -> None:
    if not condition:
        raise LegacyLabelsError(code, detail)


def _load_json(path: Path) -> dict:
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LegacyLabelsError("INPUT_UNREADABLE", f"{path}: {exc!r}") from exc
    if not isinstance(doc, dict):
        raise LegacyLabelsError("INPUT_MALFORMED", f"{path}: not a mapping")
    return doc


def build_legacy_labels(features_dir: str | Path,
                        observed_dir: str | Path,
                        out_dir: str | Path,
                        *,
                        cv_sample_states: int = CV_SAMPLE_STATES,
                        seed: int = 20261017) -> dict:
    """Export legacy label-side npz per reaction; return the audit document."""
    features_dir = Path(features_dir)
    observed_dir = Path(observed_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = _load_json(features_dir / "features_manifest.json")
    rows = manifest.get("per_reaction")
    _require(isinstance(rows, list), "MANIFEST_MALFORMED", "per_reaction missing")

    rng = np.random.default_rng(seed)
    per_reaction: list[dict] = []
    # collect a deterministic sample of (reaction, state_id) for cross-validation
    sample_pool: list[tuple[str, str]] = []
    max_abs_err = 0.0
    centroid_shift_max = 0.0
    cv_rows: list[dict] = []

    for row in rows:
        rid = str(row["reaction_group_id"])
        npz_path = features_dir / str(row["npz_file"])
        _require(npz_path.is_file(), "NPZ_MISSING", f"{npz_path} not found")
        shard = np.load(npz_path, allow_pickle=False)
        sid = [str(x) for x in shard["state_id"]]
        next_id = [str(x) for x in shard["next_state_id"]]
        is_endpoint = np.asarray(shard["is_teacher_endpoint"], dtype=bool)
        near_zero = np.asarray(shard["near_zero_action_mask"], dtype=bool)
        x = np.asarray(shard["x_angstrom"], dtype=np.float64)
        n_atoms = int(np.asarray(shard["atomic_numbers"]).size)

        teacher = DualTSTeacher.from_payload(
            _load_json(observed_dir / f"{rid}.teacher.json"))
        labels = {lab.state_id: lab for lab in teacher.labels}

        n = len(sid)
        action = np.full((n, n_atoms, 3), np.nan, dtype=np.float64)
        remaining = np.zeros(n, dtype=np.float64)
        is_terminal = np.zeros(n, dtype=bool)
        index = {s: i for i, s in enumerate(sid)}
        for i, s in enumerate(sid):
            lab = labels[s]
            is_terminal[i] = bool(is_endpoint[i])
            remaining[i] = float(lab.remaining_arc_length)
            if lab.action_delta is not None:
                action[i] = np.asarray(lab.action_delta, dtype=np.float64)
            # sanity: terminal states must have no action and vice versa
            _require((lab.action_delta is None) == bool(is_endpoint[i]),
                     "ACTION_TERMINAL_MISMATCH",
                     f"{rid} state {s}: action/terminal flag disagree")

        # centroid-shift diagnostic: |x[next]-x[curr] - action_delta| magnitude
        for i, s in enumerate(sid):
            ns = next_id[i]
            if ns == "" or action[i][0, 0] != action[i][0, 0]:  # NaN guard
                continue
            j = index.get(ns)
            if j is None:
                continue
            diff = np.asarray(x[j] - x[i], dtype=np.float64) - action[i]
            centroid_shift_max = max(
                centroid_shift_max, float(np.max(np.abs(diff))))
            sample_pool.append((rid, s))

        out_npz = {
            "state_id": np.asarray(sid),
            "next_state_id": np.asarray(next_id),
            "action_angstrom": action,
            "remaining_arc_length": remaining,
            "is_terminal": is_terminal,
            "near_zero_action_mask": near_zero,
            "is_teacher_endpoint": is_endpoint,
        }
        out_path = out_dir / f"legacy_labels_{rid}.npz"
        np.savez_compressed(out_path, **out_npz)
        per_reaction.append({
            "reaction_group_id": rid,
            "reaction_id": str(row.get("reaction_id", rid)),
            "n_states": n,
            "n_actions": int((~is_terminal).sum()),
            "npz_file": out_path.name,
            "npz_sha256": sha256_file(out_path),
        })

    # deterministic cross-validation sample of 20 states.
    sample_pool.sort()
    if len(sample_pool) > cv_sample_states:
        idx = rng.choice(len(sample_pool), size=cv_sample_states,
                         replace=False)
        idx = sorted(int(i) for i in idx)
        sample = [sample_pool[i] for i in idx]
    else:
        sample = sample_pool

    for rid, s in sample:
        shard = np.load(out_dir / f"legacy_labels_{rid}.npz",
                        allow_pickle=False)
        sid = [str(x) for x in shard["state_id"]]
        i = sid.index(s)
        act = np.asarray(shard["action_angstrom"][i], dtype=np.float64)
        teacher = DualTSTeacher.from_payload(
            _load_json(observed_dir / f"{rid}.teacher.json"))
        lab = {l.state_id: l for l in teacher.labels}[s]
        ref = np.asarray(lab.action_delta, dtype=np.float64)
        err = float(np.max(np.abs(act - ref)))
        max_abs_err = max(max_abs_err, err)
        cv_rows.append({
            "reaction_group_id": rid, "state_id": s,
            "max_abs_err": err, "passed": err <= ACTION_TOL,
        })

    _require(max_abs_err <= ACTION_TOL, "LEGACY_LABEL_MISMATCH",
             f"max |action_angstrom - teacher.action_delta| = {max_abs_err:.3e} "
             f"> {ACTION_TOL:g}")

    audit = {
        "schema_version": LEGACY_LABELS_SCHEMA,
        "tool_version": LEGACY_LABELS_TOOL_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "features_dir": str(features_dir),
        "observed_dir": str(observed_dir),
        "out_dir": str(out_dir),
        "action_definition": ("action_angstrom = teacher label.action_delta "
                              "(registered X_next - X); NaN at terminal"),
        "centroid_note": ("shard x_angstrom is centroid-removed (frozen "
                          "consume-time rule), so x[next]-x[curr] differs from "
                          "action_delta by the per-frame centroid shift; the "
                          "label is taken from the registered teacher frame"),
        "centroid_shift_max_abs_angstrom": centroid_shift_max,
        "cross_validation": {
            "n_states": len(cv_rows),
            "tolerance": ACTION_TOL,
            "max_abs_err": max_abs_err,
            "passed": max_abs_err <= ACTION_TOL,
            "rows": cv_rows,
        },
        "per_reaction": sorted(per_reaction,
                               key=lambda r: r["reaction_group_id"]),
    }
    audit_path = out_dir / "legacy_labels_audit.json"
    audit_path.write_text(json.dumps(audit, indent=1, sort_keys=False,
                                     ensure_ascii=False, allow_nan=False),
                          encoding="utf-8")
    return audit


__all__ = [
    "ACTION_TOL",
    "CV_SAMPLE_STATES",
    "LEGACY_LABELS_SCHEMA",
    "LEGACY_LABELS_TOOL_VERSION",
    "LegacyLabelsError",
    "build_legacy_labels",
]
