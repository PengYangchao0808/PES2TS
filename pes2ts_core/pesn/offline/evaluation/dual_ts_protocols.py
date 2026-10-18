"""Lane D — V4.2–V4.7 three-protocol evaluation freeze + metrics (contract §7, plan §10).

Three evaluation "protocols" (口径) are frozen BEFORE any trained-model
prediction is seen, then every arm is scored identically against them:

* ``legacy_full``   — the frozen monitor8 v3 features (all subdivided
  actions), the ORIGINAL group roster, and the original direction / near-TS
  masks.  Keeps the historical four-gate regression check.
* ``native_full``   — the native reconnected monitor8 features (all native
  actions, all 8 groups).  New-task primary metrics; the four gate
  thresholds (1.0 / 6-of-8 / 0.30 / 0.40) are pre-registered REUSED values
  but explicitly flagged as a NEW protocol (never rewrites historical G5).
* ``common_unchanged`` — the intersection transfer set: a transfer whose
  current frame ``state_id`` exists in BOTH libraries, was NOT subdivided
  (its legacy successor IS its native successor), and whose action / mask
  agree to |Δaction| ≤ 1e-10.  A same-question comparison; no four-gate is
  applied, only the difference and the missing denominators are reported.

The freeze artifact is content-digested and write-once (``FREEZE_EXISTS`` on
any re-freeze).  The evaluation core is numpy-only (no torch, no project
imports) so it runs on CPU synthetic data and can be reused by Lane C.

Public surface:

* :func:`freeze_protocols` — build + write the freeze, return it (+ in-memory
  ``_reference`` transfers).
* :func:`load_protocol_reference` — read-only reload of the same freeze
  (for the evaluation script, after the write-once freeze already exists).
* :func:`verify_freeze_payload` — re-digest a (possibly disk-loaded) payload.
* :func:`evaluate_protocol` — full metric block for one protocol.
* :func:`apply_gates` — the four-gate decision for legacy/native; no gate for
  common_unchanged.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, Sequence

import numpy as np

PROTOCOL_FREEZE_SCHEMA: Final[str] = "pesn_v42_campaign_protocol_v1"
PROTOCOL_EVAL_SCHEMA: Final[str] = "pesn_v42_protocol_eval_v1"

# ---------------------------------------------------------------------------
# Frozen gate thresholds (literal source: scripts/freeze_wp4_monitor_prereg.py
# freeze_thresholds() -> payload["gate"]).
# ---------------------------------------------------------------------------
GATE_MEDIAN_RATIO_MAX: Final[float] = 1.0
GATE_N_GROUPS_RATIO_BELOW_ONE_MIN: Final[int] = 6
GATE_MEDIAN_COSINE_MIN: Final[float] = 0.30
GATE_NEAR_TS_COSINE_MIN: Final[float] = 0.40
GATE_N_GROUPS: Final[int] = 8

DEFAULT_GATE_THRESHOLDS: Final[dict] = {
    "median_mse_ratio_max": GATE_MEDIAN_RATIO_MAX,
    "n_groups_ratio_below_one_min": GATE_N_GROUPS_RATIO_BELOW_ONE_MIN,
    "median_cosine_min": GATE_MEDIAN_COSINE_MIN,
    "near_ts_cosine_min": GATE_NEAR_TS_COSINE_MIN,
    "n_groups": GATE_N_GROUPS,
}

#: Frozen near-zero direction mask threshold (dual_ts_spec
#: ``NEAR_ZERO_ACTION_THRESHOLD_ANGSTROM``).
NEAR_ZERO_THRESHOLD_ANGSTROM: Final[float] = 0.01
#: Frozen near-TS arc rule (dual_ts_sampling ``NEAR_TS_ARC_FRACTION``).
NEAR_TS_ARC_FRACTION: Final[float] = 0.2
#: Contract §2 label-identity tolerance (float64 exact re-derivation).
ACTION_EPS: Final[float] = 1e-10

PROTOCOL_NAMES: Final[tuple[str, ...]] = (
    "legacy_full", "native_full", "common_unchanged")

LEGACY_FULL_IDENTITY: Final[str] = (
    "monitor8 v3 features (spec pesn_dual_ts_features_v3): ALL subdivided "
    "actions, original group roster, original direction/near-TS masks; "
    "historical four-gate regression check")
NATIVE_FULL_IDENTITY: Final[str] = (
    "native reconnected monitor8 features (pesn_dual_ts_features_native_v1): "
    "ALL native actions, all 8 groups; new-task primary metrics; four-gate "
    "thresholds pre-registered REUSED but flagged new_protocol")
COMMON_UNCHANGED_IDENTITY: Final[str] = (
    "intersection transfer set: current frame in both libraries, not "
    "subdivided (legacy successor == native successor), action/mask agree "
    "to |Δaction| <= 1e-10; same-question comparison, no four-gate")


class ProtocolError(RuntimeError):
    """Typed protocol freeze / evaluation refusal (machine-readable code)."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json_sha256(obj) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, allow_nan=False)
        .encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _branch_label(branch_id: str) -> str:
    """Normalize a branch id to its endpoint label (A/B) when parseable."""
    s = str(branch_id)
    if "#br" in s:
        pos = s.rfind("#br") + 3
        if pos < len(s):
            return s[pos].upper()
    if s and s[-1] in ("A", "B", "a", "b"):
        return s[-1].upper()
    return s


def _median(values: Sequence[float]) -> float | None:
    finite = [float(v) for v in values
              if v is not None and math.isfinite(float(v))]
    return float(np.median(finite)) if finite else None


def _derived_near_zero(action: np.ndarray, is_terminal: np.ndarray) -> np.ndarray:
    """Recompute the frozen near-zero mask (‖a_i‖ < 0.01) when not stored."""
    n_states, n_atoms = action.shape[0], action.shape[1]
    out = np.zeros((n_states, n_atoms), dtype=bool)
    for r in range(n_states):
        if is_terminal[r]:
            out[r] = True
        else:
            out[r] = np.linalg.norm(action[r], axis=1) < NEAR_ZERO_THRESHOLD_ANGSTROM
    return out


def _critical_pairs(shard: dict) -> list[tuple[int, int]]:
    """Union-edge atom pairs where either endpoint bond table has changed_flag=1.

    The union edge set ``edge_index`` is directed (both orientations); the
    changed flag is channel 2 of ``edge_feats_a``/``edge_feats_b`` (plan §10:
    "关键对白名单 = 任一端点 bond 表 changed_flag=1 的并集边").  Missing /
    malformed edge data yields an empty list (honest absence, never a guess).
    """
    for key in ("edge_index", "edge_feats_a", "edge_feats_b"):
        if key not in shard:
            return []
    ei = np.asarray(shard["edge_index"])
    fa = np.asarray(shard["edge_feats_a"])
    fb = np.asarray(shard["edge_feats_b"])
    if ei.ndim != 2 or ei.shape[0] != 2 or fa.ndim != 2 or fb.ndim != 2:
        return []
    if fa.shape[1] < 3 or fb.shape[1] < 3:
        return []
    changed = (fa[:, 2] > 0.5) | (fb[:, 2] > 0.5)
    pairs = set()
    for e in np.where(changed)[0]:
        i, j = int(ei[0, e]), int(ei[1, e])
        if i == j:
            continue
        pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


# ---------------------------------------------------------------------------
# Arm loaders (legacy npz+teacher.json / native npz)
# ---------------------------------------------------------------------------

def _resolve_observed_dir(legacy_dir: Path, manifest: dict) -> Path:
    sibling = legacy_dir.parent / "teachers_observed"
    if sibling.is_dir():
        return sibling
    observed = manifest.get("observed_dir")
    if observed:
        rel = Path(str(observed))
        bases = [legacy_dir.parent, legacy_dir.parent.parent, Path.cwd()]
        for base in bases:
            for cand in (base / rel, base / rel.name):
                if cand.is_dir():
                    return cand
    raise ProtocolError(
        "OBSERVED_DIR_MISSING",
        f"cannot resolve the observed teacher dir for legacy features at "
        f"{legacy_dir} (looked for teachers_observed sibling + manifest "
        f"observed_dir)")


def _load_legacy_transfers(legacy_dir: Path) -> tuple[list[dict], list[str], str]:
    """Legacy monitor8 v3 transfers: npz inputs + teacher label actions.

    Returns ``(transfers, group_ids, spec_digest)`` where each transfer is an
    in-memory dict carrying numpy arrays (never serialized into the freeze).
    """
    manifest = json.loads((legacy_dir / "features_manifest.json").read_text())
    observed_dir = _resolve_observed_dir(legacy_dir, manifest)
    spec_digest = str(manifest.get("spec_digest", ""))
    transfers: list[dict] = []
    group_ids: list[str] = []
    for row in sorted(manifest.get("per_reaction", []),
                      key=lambda r: str(r.get("reaction_group_id"))):
        shard = np.load(legacy_dir / row["npz_file"])
        rid = str(shard["reaction_id"].item())
        gid = str(shard["reaction_group_id"].item())
        teacher = json.loads((observed_dir / f"{rid}.teacher.json").read_text())
        labels = {str(l["state_id"]): l for l in teacher["labels"]}
        z = np.asarray(shard["atomic_numbers"], dtype=int)
        x = np.asarray(shard["x_angstrom"], dtype=np.float64)
        near_zero = np.asarray(shard["near_zero_action_mask"], dtype=bool)
        is_teacher = np.asarray(shard["is_teacher_endpoint"], dtype=bool)
        state_ids = [str(s) for s in shard["state_id"]]
        next_ids = [str(s) for s in shard["next_state_id"]]
        branch_ids = [str(b) for b in shard["branch_id"]]
        crit = _critical_pairs(shard)
        for r, sid in enumerate(state_ids):
            lab = labels[sid]
            action = (None if lab.get("action_delta") is None
                      else np.asarray(lab["action_delta"], dtype=np.float64))
            transfers.append({
                "state_id": sid,
                "group_id": gid,
                "branch_id": branch_ids[r],
                "next_state_id": next_ids[r],
                "n_atoms": int(z.size),
                "atomic_numbers": z,
                "action": action,
                "near_zero_mask": near_zero[r].copy(),
                "is_terminal": bool(is_teacher[r]),
                "remaining_arc_length": float(lab["remaining_arc_length"]),
                "x_angstrom": x[r],
                "critical_pairs": crit,
            })
        group_ids.append(gid)
    return transfers, group_ids, spec_digest


def _load_native_transfers(native_dir: Path) -> tuple[list[dict], list[str], str]:
    """Native monitor8 reconnected transfers (Lane A ``native_monitor_features``).

    Expected per-npz arrays: ``state_id``, ``next_state_id``, ``branch_id``,
    ``atomic_numbers``, ``x_angstrom``, ``action_angstrom`` (S,N,3),
    ``is_terminal`` (S,), ``remaining_arc_length`` (S,) and (optionally) the
    inherited ``near_zero_action_mask`` (derived from ``action_angstrom`` with
    the frozen 0.01 Å rule when absent).
    """
    manifest = json.loads((native_dir / "features_manifest.json").read_text())
    spec_digest = str(manifest.get("spec_digest", ""))
    transfers: list[dict] = []
    group_ids: list[str] = []
    for row in sorted(manifest.get("per_reaction", []),
                      key=lambda r: str(r.get("reaction_group_id"))):
        shard = np.load(native_dir / row["npz_file"])
        gid = str(shard["reaction_group_id"].item())
        z = np.asarray(shard["atomic_numbers"], dtype=int)
        x = np.asarray(shard["x_angstrom"], dtype=np.float64)
        action = np.asarray(shard["action_angstrom"], dtype=np.float64)
        is_terminal = np.asarray(shard["is_terminal"], dtype=bool)
        ral = np.asarray(shard["remaining_arc_length"], dtype=np.float64)
        if "near_zero_action_mask" in shard:
            near_zero = np.asarray(shard["near_zero_action_mask"], dtype=bool)
        else:
            near_zero = _derived_near_zero(action, is_terminal)
        state_ids = [str(s) for s in shard["state_id"]]
        next_ids = [str(s) for s in shard["next_state_id"]]
        branch_ids = [str(b) for b in shard["branch_id"]]
        crit = _critical_pairs(shard)
        for r, sid in enumerate(state_ids):
            a = (None if is_terminal[r] else action[r].copy())
            transfers.append({
                "state_id": sid,
                "group_id": gid,
                "branch_id": branch_ids[r],
                "next_state_id": next_ids[r],
                "n_atoms": int(z.size),
                "atomic_numbers": z,
                "action": a,
                "near_zero_mask": near_zero[r].copy(),
                "is_terminal": bool(is_terminal[r]),
                "remaining_arc_length": float(ral[r]),
                "x_angstrom": x[r],
                "critical_pairs": crit,
            })
        group_ids.append(gid)
    return transfers, group_ids, spec_digest


def _actions_equal(lt: dict, nt: dict) -> bool:
    la, na = lt["action"], nt["action"]
    if la is None or na is None:
        return la is None and na is None
    la = np.asarray(la, dtype=np.float64)
    na = np.asarray(na, dtype=np.float64)
    if la.shape != na.shape:
        return False
    return bool(np.max(np.abs(la - na)) <= ACTION_EPS)


def _compute_common_unchanged(legacy: list[dict],
                              native: list[dict]) -> list[dict]:
    """The intersection transfer set (plan §10 / contract §7 common_unchanged)."""
    native_by_id = {t["state_id"]: t for t in native}
    common: list[dict] = []
    for lt in legacy:
        if lt["is_terminal"]:
            continue  # teacher endpoints carry no outgoing transfer
        nt = native_by_id.get(lt["state_id"])
        if nt is None:
            continue  # subdivided (inserted) state — not in native library
        if lt["next_state_id"] != nt["next_state_id"]:
            continue  # subdivided PREDECESSOR — legacy successor was an insert
        if not _actions_equal(lt, nt):
            continue
        if not bool(np.array_equal(np.asarray(lt["near_zero_mask"], dtype=bool),
                                   np.asarray(nt["near_zero_mask"], dtype=bool))):
            continue
        common.append(lt)  # legacy record (agrees with native by construction)
    common.sort(key=lambda t: t["state_id"])
    return common


# ---------------------------------------------------------------------------
# Counts / denominators
# ---------------------------------------------------------------------------

def _branch_max(transfers: list[dict]) -> dict[tuple[str, str], float]:
    out: dict[tuple[str, str], float] = {}
    for t in transfers:
        if t["is_terminal"] or t["action"] is None:
            continue
        key = (t["group_id"], t["branch_id"])
        out[key] = max(out.get(key, 0.0), float(t["remaining_arc_length"]))
    return out


def _per_group_counts(transfers: list[dict],
                      group_ids: Sequence[str]) -> dict[str, dict]:
    bmax = _branch_max(transfers)
    per_group: dict[str, dict] = {}
    for g in group_ids:
        per_group[str(g)] = {
            "n_transfers": 0, "n_action_states": 0,
            "n_direction_atoms": 0, "n_near_zero_atoms": 0,
            "n_near_ts_states": 0,
        }
    for t in transfers:
        g = str(t["group_id"])
        pg = per_group.setdefault(g, dict(
            n_transfers=0, n_action_states=0, n_direction_atoms=0,
            n_near_zero_atoms=0, n_near_ts_states=0))
        pg["n_transfers"] += 1
        if t["is_terminal"] or t["action"] is None:
            continue
        nz = np.asarray(t["near_zero_mask"], dtype=bool)
        pg["n_action_states"] += 1
        pg["n_direction_atoms"] += int((~nz).sum())
        pg["n_near_zero_atoms"] += int(nz.sum())
        if float(t["remaining_arc_length"]) <= NEAR_TS_ARC_FRACTION * \
                bmax[(g, t["branch_id"])]:
            pg["n_near_ts_states"] += 1
    return per_group


def _totals(per_group: dict[str, dict]) -> dict:
    keys = ("n_transfers", "n_action_states", "n_direction_atoms",
            "n_near_zero_atoms", "n_near_ts_states")
    return {k: int(sum(pg[k] for pg in per_group.values())) for k in keys}


def _protocol_counts(transfers: list[dict],
                     group_ids: Sequence[str]) -> dict:
    per_group = _per_group_counts(transfers, group_ids)
    return {"denominator": _totals(per_group), "per_group": per_group}


def _missing_denominators(common: list[dict],
                          native_group_ids: Sequence[str],
                          legacy_group_ids: Sequence[str]) -> dict:
    """Groups whose unchanged subset lost direction / near-TS samples.

    Plan §10: an intersection that drops a group's effective direction or
    near-TS samples must be recorded (missing denominator), never zero-filled
    or dressed up as a full 8-group gate.
    """
    per_group = _per_group_counts(common, native_group_ids)
    missing = {}
    for g, pg in per_group.items():
        problems = []
        if pg["n_action_states"] == 0:
            problems.append("no_unchanged_action_states")
        if pg["n_direction_atoms"] == 0:
            problems.append("no_unchanged_direction_atoms")
        if pg["n_near_ts_states"] == 0:
            problems.append("no_unchanged_near_ts_states")
        if problems:
            missing[g] = {"problems": problems, "counts": pg}
    return {
        "note": "groups whose common_unchanged subset lacks direction or "
                "near-TS samples (never zero-filled, never counted as a full "
                "8-group gate)",
        "missing_groups": missing,
        "n_groups_with_missing": len(missing),
        "legacy_n_groups": len(legacy_group_ids),
        "native_n_groups": len(native_group_ids),
        "common_n_groups": len({t["group_id"] for t in common}),
    }


# ---------------------------------------------------------------------------
# Freeze
# ---------------------------------------------------------------------------

def _build_freeze_payload(legacy_dir: Path,
                          native_dir: Path) -> tuple[dict, dict]:
    legacy, legacy_groups, legacy_spec = _load_legacy_transfers(legacy_dir)
    native, native_groups, native_spec = _load_native_transfers(native_dir)
    common = _compute_common_unchanged(legacy, native)

    def proto_def(transfers, group_ids, identity):
        counts = _protocol_counts(transfers, group_ids)
        return {
            "identity": identity,
            "group_ids": [str(g) for g in group_ids],
            "n_groups": len(group_ids),
            "denominator": counts["denominator"],
            "per_group": counts["per_group"],
        }

    legacy_def = proto_def(legacy, legacy_groups, LEGACY_FULL_IDENTITY)
    native_def = proto_def(native, native_groups, NATIVE_FULL_IDENTITY)
    common_counts = _protocol_counts(common, native_groups)
    common_def = {
        "identity": COMMON_UNCHANGED_IDENTITY,
        "roster": sorted(t["state_id"] for t in common),
        "n_transfers": len(common),
        "n_groups": len({t["group_id"] for t in common}),
        "denominator": common_counts["denominator"],
        "per_group_counts": common_counts["per_group"],
        "missing": _missing_denominators(common, native_groups, legacy_groups),
    }

    parent_hashes = {
        "legacy_manifest_sha256": _sha256_file(legacy_dir / "features_manifest.json"),
        "legacy_spec_digest": legacy_spec,
        "native_manifest_sha256": _sha256_file(native_dir / "features_manifest.json"),
        "native_spec_digest": native_spec,
    }
    deterministic_core = {
        "schema_version": PROTOCOL_FREEZE_SCHEMA,
        "parent_hashes": parent_hashes,
        "freeze_rule": (
            "legacy_full = legacy 8 groups ALL subdivided actions + original "
            "roster + direction/near-TS masks; native_full = native 8 groups "
            "ALL native actions; common_unchanged = transfers whose current "
            "frame exists in both libraries, is not subdivided (legacy "
            "successor == native successor), and action/mask agree to "
            "|Δaction| <= 1e-10.  Frozen before any model prediction; "
            "write-once."),
        "protocols": {
            "legacy_full": legacy_def,
            "native_full": native_def,
            "common_unchanged": common_def,
        },
    }
    payload = {
        "schema_version": PROTOCOL_FREEZE_SCHEMA,
        "title": "V4.2–V4.7 campaign three-protocol evaluation freeze "
                 "(contract §7; frozen before any model prediction)",
        "frozen_utc": _utc_now(),
        **deterministic_core,
        "payload_sha256": _json_sha256(deterministic_core),
    }
    reference = {
        "legacy_full": legacy,
        "native_full": native,
        "common_unchanged": common,
    }
    return payload, reference


def freeze_protocols(legacy_dir: str | Path, native_dir: str | Path,
                     out_path: str | Path) -> dict:
    """Freeze the three protocols; write ``out_path`` once, refuse to rewrite.

    Returns the freeze payload with an in-memory ``_reference`` key carrying
    the per-protocol transfer records (numpy arrays; NOT serialized).  The
    on-disk JSON is the durable content-digested identity.
    """
    legacy_dir = Path(legacy_dir)
    native_dir = Path(native_dir)
    out_path = Path(out_path)
    if out_path.exists():
        raise ProtocolError(
            "FREEZE_EXISTS",
            f"{out_path} already exists — the protocol freeze is written "
            f"once and must never be rewritten (freeze a new version under a "
            f"new filename)")
    payload, reference = _build_freeze_payload(legacy_dir, native_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {**payload, "_reference": reference}


def load_protocol_reference(legacy_dir: str | Path,
                            native_dir: str | Path) -> dict:
    """Read-only reload of the freeze payload + reference transfers.

    Used by the evaluation script AFTER the write-once freeze already exists:
    it re-loads the reference arrays and recomputes the deterministic digest
    so the caller can assert it matches the on-disk ``protocol_freeze.json``.
    """
    payload, reference = _build_freeze_payload(Path(legacy_dir),
                                               Path(native_dir))
    return {**payload, "_reference": reference}


def freeze_digest(payload: dict) -> str:
    """Recompute the deterministic digest of a (possibly disk-loaded) payload."""
    core = {
        "schema_version": payload.get("schema_version"),
        "parent_hashes": payload.get("parent_hashes"),
        "freeze_rule": payload.get("freeze_rule"),
        "protocols": payload.get("protocols"),
    }
    return _json_sha256(core)


def verify_freeze_payload(payload: dict) -> dict:
    """Return ``{ok, recorded, recomputed}`` for a freeze payload identity."""
    recorded = str(payload.get("payload_sha256", ""))
    recomputed = freeze_digest(payload)
    return {"ok": bool(recorded and recorded == recomputed),
            "recorded": recorded, "recomputed": recomputed}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _require_ref(frozen: dict, protocol: str) -> list[dict]:
    if protocol not in PROTOCOL_NAMES:
        raise ProtocolError("PROTOCOL_UNKNOWN",
                            f"protocol {protocol!r} not in {PROTOCOL_NAMES}")
    ref = frozen.get("_reference")
    if not isinstance(ref, dict) or protocol not in ref:
        raise ProtocolError(
            "REFERENCE_MISSING",
            f"frozen payload has no in-memory _reference for {protocol!r}; "
            f"reload via load_protocol_reference(legacy_dir, native_dir)")
    return ref[protocol]


def evaluate_protocol(pred_actions_by_state: dict, protocol: str,
                      frozen: dict) -> dict:
    """Full metric block for one protocol (plan §10 field list).

    ``pred_actions_by_state`` maps ``state_id -> (N,3)`` predicted actions.
    Predictions are shape-validated against the reference (no silent
    truncation); a missing or mis-shaped prediction raises a typed
    :class:`ProtocolError`.
    """
    transfers = _require_ref(frozen, protocol)
    bmax = _branch_max(transfers)

    groups = sorted({t["group_id"] for t in transfers})
    # per-atom accumulators
    g_a: list[str] = []
    bl_a: list[str] = []
    nt_a: list[bool] = []
    dir_a: list[bool] = []
    nz_a: list[bool] = []
    crit_a: list[bool] = []
    z_a: list[int] = []
    anorm_a: list[float] = []
    mse_a: list[float] = []
    zero_a: list[float] = []
    cos_a: list[float] = []
    alpha_a: list[float] = []
    crit_rows: list[tuple[float, float, float]] = []  # (err, dd_true, dd_pred)
    n_action_states = 0
    n_near_ts_states = 0

    for t in transfers:
        if t["is_terminal"] or t["action"] is None:
            continue
        n_action_states += 1
        sid = t["state_id"]
        pred = pred_actions_by_state.get(sid)
        if pred is None:
            raise ProtocolError("MISSING_PREDICTION",
                                f"no prediction supplied for state {sid}")
        pred = np.asarray(pred, dtype=np.float64)
        a = np.asarray(t["action"], dtype=np.float64)
        if pred.shape != a.shape or pred.ndim != 2 or pred.shape[1] != 3:
            raise ProtocolError(
                "SHAPE_MISMATCH",
                f"state {sid}: prediction {pred.shape} != reference {a.shape} "
                f"(no silent truncation)")
        nz = np.asarray(t["near_zero_mask"], dtype=bool)
        if nz.shape != (a.shape[0],):
            raise ProtocolError("MASK_MISMATCH",
                                f"state {sid}: near-zero mask shape mismatch")
        d = pred - a
        mse = (d * d).sum(1)
        zero = (a * a).sum(1)
        anorm = np.linalg.norm(a, axis=1)
        pnorm = np.linalg.norm(pred, axis=1)
        dirmask = ~nz
        with np.errstate(divide="ignore", invalid="ignore"):
            cos = np.where(dirmask,
                           (pred * a).sum(1) / np.maximum(pnorm * anorm, 1e-12),
                           0.0)
            alpha = np.where(dirmask, pnorm / np.maximum(anorm, 1e-12), np.nan)
        near_ts = float(t["remaining_arc_length"]) <= \
            NEAR_TS_ARC_FRACTION * bmax[(t["group_id"], t["branch_id"])]
        n_near_ts_states += int(near_ts)
        g = t["group_id"]
        bl = _branch_label(t["branch_id"])
        z = np.asarray(t["atomic_numbers"], dtype=int)
        crit_set = set()
        for (i, j) in t["critical_pairs"]:
            i, j = int(i), int(j)
            if 0 <= i < a.shape[0] and 0 <= j < a.shape[0]:
                crit_set.add(i)
                crit_set.add(j)
        crit_flag = np.zeros(a.shape[0], dtype=bool)
        for i in crit_set:
            crit_flag[i] = True

        n = a.shape[0]
        g_a.extend([g] * n)
        bl_a.extend([bl] * n)
        nt_a.extend([bool(near_ts)] * n)
        dir_a.extend(dirmask.tolist())
        nz_a.extend(nz.tolist())
        crit_a.extend(crit_flag.tolist())
        z_a.extend(z.tolist())
        anorm_a.extend(anorm.tolist())
        mse_a.extend(mse.tolist())
        zero_a.extend(zero.tolist())
        cos_a.extend(cos.tolist())
        alpha_a.extend(alpha.tolist())

        x = np.asarray(t["x_angstrom"], dtype=np.float64)
        for (i, j) in t["critical_pairs"]:
            i, j = int(i), int(j)
            if i >= a.shape[0] or j >= a.shape[0] or i == j:
                continue
            d_before = float(np.linalg.norm(x[i] - x[j]))
            d_true = float(np.linalg.norm((x[i] + a[i]) - (x[j] + a[j])))
            d_pred = float(np.linalg.norm((x[i] + pred[i]) - (x[j] + pred[j])))
            crit_rows.append((abs(d_pred - d_true), d_true - d_before,
                              d_pred - d_before))

    # ---- vectorize accumulators ----
    mse = np.asarray(mse_a, dtype=np.float64)
    zero = np.asarray(zero_a, dtype=np.float64)
    anorm = np.asarray(anorm_a, dtype=np.float64)
    cos = np.asarray(cos_a, dtype=np.float64)
    alpha = np.asarray(alpha_a, dtype=np.float64)
    dir_m = np.asarray(dir_a, dtype=bool)
    nz_m = np.asarray(nz_a, dtype=bool)
    nt_m = np.asarray(nt_a, dtype=bool)
    crit_m = np.asarray(crit_a, dtype=bool)
    z = np.asarray(z_a, dtype=int)
    g_arr = np.asarray(g_a)
    bl_arr = np.asarray(bl_a)

    n_atoms = int(mse.size)
    denominator = {
        "n_transfers": len(transfers),
        "n_action_states": n_action_states,
        "n_atoms_total": n_atoms,
        "n_direction_atoms": int(dir_m.sum()),
        "n_near_zero_atoms": int(nz_m.sum()),
        "n_near_ts_states": n_near_ts_states,
        "n_groups": len(groups),
        "missing_note": (frozen.get("protocols", {}).get(protocol, {})
                         .get("missing", {}).get("note"))
        if protocol == "common_unchanged" else None,
    }

    # ---- per-group aggregation ----
    per_group_mse: dict[str, float] = {}
    per_group_zero: dict[str, float] = {}
    per_group_ratio: dict[str, float] = {}
    per_group_cos: dict[str, float | None] = {}
    per_group_nt_cos: dict[str, float | None] = {}
    for g in groups:
        gm = g_arr == g
        if int(gm.sum()) == 0:
            per_group_mse[g] = float("nan")
            per_group_zero[g] = float("nan")
            per_group_ratio[g] = float("nan")
            per_group_cos[g] = None
            per_group_nt_cos[g] = None
            continue
        per_group_mse[g] = float(mse[gm].mean())
        per_group_zero[g] = float(zero[gm].mean())
        per_group_ratio[g] = (per_group_mse[g] / per_group_zero[g]
                              if per_group_zero[g] > 0 else float("inf"))
        dm = gm & dir_m
        per_group_cos[g] = _weighted_cos(cos[dm], anorm[dm])
        dnt = gm & dir_m & nt_m
        per_group_nt_cos[g] = _weighted_cos(cos[dnt], anorm[dnt])

    ratios = [per_group_ratio[g] for g in groups if math.isfinite(per_group_ratio[g])]
    coses = [per_group_cos[g] for g in groups if per_group_cos[g] is not None]
    nt_coses = [per_group_nt_cos[g] for g in groups
                if per_group_nt_cos[g] is not None]

    def weighted_cos(mask):
        return _weighted_cos(cos[mask], anorm[mask])

    def slice_mse(mask):
        return float(mse[mask].mean()) if int(mask.sum()) else float("nan")

    # ---- slices ----
    long_p90 = float(np.quantile(anorm, 0.90)) if anorm.size else float("nan")
    long_p99 = float(np.quantile(anorm, 0.99)) if anorm.size else float("nan")
    h_mask = z == 1
    heavy_mask = z > 1
    branch_a = bl_arr == "A"
    branch_b = bl_arr == "B"
    long90_mask = anorm >= long_p90
    long99_mask = anorm >= long_p99

    alpha_defined = alpha[~np.isnan(alpha)]

    crit_err = np.asarray([r[0] for r in crit_rows]) if crit_rows \
        else np.asarray([], dtype=np.float64)

    metrics = {
        "schema_version": PROTOCOL_EVAL_SCHEMA,
        "protocol": protocol,
        "denominator": denominator,
        "mse": {
            "per_group_mse_A2": {g: per_group_mse[g] for g in groups},
            "per_group_zero_A2": {g: per_group_zero[g] for g in groups},
            "per_group_ratio_vs_zero": {g: per_group_ratio[g] for g in groups},
            "group_equal_weighted": {
                "mse_A2": float(np.nanmean([per_group_mse[g] for g in groups]))
                if groups else float("nan"),
                "ratio_mean": float(np.nanmean([per_group_ratio[g]
                                                for g in groups]))
                if groups else float("nan"),
                "ratio_median": float(np.median(ratios)) if ratios
                else float("nan"),
                "ratio_below_one_groups": int(
                    sum(1 for v in ratios if v < 1.0)),
                "n_groups": len(groups),
            },
            "atom_pooled": {
                "mse_A2": float(mse.mean()) if mse.size else float("nan"),
                "zero_A2": float(zero.mean()) if zero.size else float("nan"),
                "ratio_vs_zero": (float(mse.mean() / zero.mean())
                                  if mse.size and zero.mean() > 0
                                  else float("nan")),
            },
        },
        "direction": {
            "overall_cosine": weighted_cos(dir_m),
            "n_direction_atoms": int(dir_m.sum()),
            "cosine_median": float(np.median(coses)) if coses else None,
            "per_group_cosine": {g: per_group_cos[g] for g in groups},
        },
        "near_ts": {
            "cosine": weighted_cos(dir_m & nt_m),
            "n_direction_atoms": int((dir_m & nt_m).sum()),
            "cosine_median": float(np.median(nt_coses)) if nt_coses else None,
            "per_group_cosine": {g: per_group_nt_cos[g] for g in groups},
        },
        "near_zero": {
            "abs_rmse_A": (float(np.sqrt((mse[nz_m]).mean()))
                           if int(nz_m.sum()) else None),
            "n_atoms": int(nz_m.sum()),
            "per_group_abs_rmse_A": _per_group_rmse(g_arr, groups, mse, nz_m),
        },
        "alpha": {
            "mean": float(alpha_defined.mean()) if alpha_defined.size else None,
            "median": float(np.median(alpha_defined))
            if alpha_defined.size else None,
            "p10": float(np.quantile(alpha_defined, 0.10))
            if alpha_defined.size else None,
            "p90": float(np.quantile(alpha_defined, 0.90))
            if alpha_defined.size else None,
            "n_atoms": int(alpha_defined.size),
            "near_zero_undefined_note": (
                "alpha = |a_hat|/|a*| is explicitly UNDEFINED for near-zero "
                "truth atoms (|a*| < 0.01 A); no small-denominator extremes "
                "are manufactured (plan §10)"),
        },
        "slices": {
            "H_atoms": {"n_atoms": int(h_mask.sum()),
                        "mse_A2": slice_mse(h_mask),
                        "cosine": weighted_cos(h_mask & dir_m)},
            "heavy_atoms": {"n_atoms": int(heavy_mask.sum()),
                            "mse_A2": slice_mse(heavy_mask),
                            "cosine": weighted_cos(heavy_mask & dir_m)},
            "long_step_p90": {"threshold_A": long_p90,
                              "n_atoms": int(long90_mask.sum()),
                              "mse_A2": slice_mse(long90_mask),
                              "cosine": weighted_cos(long90_mask & dir_m)},
            "long_step_p99": {"threshold_A": long_p99,
                              "n_atoms": int(long99_mask.sum()),
                              "mse_A2": slice_mse(long99_mask),
                              "cosine": weighted_cos(long99_mask & dir_m)},
            "branch_A": {"n_atoms": int(branch_a.sum()),
                         "mse_A2": slice_mse(branch_a),
                         "cosine": weighted_cos(branch_a & dir_m)},
            "branch_B": {"n_atoms": int(branch_b.sum()),
                         "mse_A2": slice_mse(branch_b),
                         "cosine": weighted_cos(branch_b & dir_m)},
            "critical_pairs": {
                "n_pairs": int(crit_rows.__len__()),
                "n_atoms": int(crit_m.sum()),
                "distance_error_mae_A": (float(crit_err.mean())
                                         if crit_err.size else None),
                "distance_error_rmse_A": (
                    float(np.sqrt((crit_err ** 2).mean()))
                    if crit_err.size else None),
                "mse_A2": slice_mse(crit_m),
                "cosine": weighted_cos(crit_m & dir_m),
                "distance_change": _distance_change_sign(crit_rows),
            },
            "spectator_atoms": {"n_atoms": int((~crit_m).sum()),
                                "mse_A2": slice_mse(~crit_m),
                                "cosine": weighted_cos((~crit_m) & dir_m)},
        },
    }
    if protocol == "common_unchanged":
        metrics["missing"] = frozen.get("protocols", {}).get(
            "common_unchanged", {}).get("missing", {})
    return metrics


def _weighted_cos(cos: np.ndarray, w: np.ndarray) -> float | None:
    if int(w.size) == 0 or float(w.sum()) <= 0.0:
        return None
    return float((w * cos).sum() / w.sum())


def _per_group_rmse(g_arr, groups, mse, mask):
    out: dict[str, float | None] = {}
    for g in groups:
        gm = (g_arr == g) & mask
        out[g] = float(np.sqrt((mse[gm]).mean())) if int(gm.sum()) else None
    return out


def _distance_change_sign(crit_rows: list[tuple[float, float, float]],
                          threshold_A: float | None = None) -> dict:
    """Distance-change sign accuracy over critical pairs.

    ``threshold_A`` parameterizes which changes count (a pair is
    "sign-eligible" when |Δd_true| > threshold).  Default is the fit-frozen
    value; the caller can override to any frozen Å threshold.
    """
    if threshold_A is None:
        threshold_A = _DEFAULT_DISTANCE_CHANGE_THRESHOLD_A
    eligible = [(dd_true, dd_pred) for (_, dd_true, dd_pred) in crit_rows
                if abs(dd_true) > threshold_A]
    if not eligible:
        return {
            "threshold_A": float(threshold_A),
            "n_eligible": 0,
            "sign_accuracy": None,
            "note": "no critical pair crosses the distance-change threshold",
        }
    correct = sum(1 for (dd_t, dd_p) in eligible
                  if (dd_t > 0) == (dd_p > 0))
    return {
        "threshold_A": float(threshold_A),
        "n_eligible": len(eligible),
        "sign_accuracy": float(correct / len(eligible)),
    }


#: Fit-frozen distance-change sign threshold (Å).  Recorded here as a
#: parameterizable default — the plan (§10) requires the threshold be frozen
#: on fit and passed in; the literal is overridable by the caller.
_DEFAULT_DISTANCE_CHANGE_THRESHOLD_A: float = 0.05


# ---------------------------------------------------------------------------
# Four-gate application
# ---------------------------------------------------------------------------

def apply_gates(metrics: dict, thresholds: dict | None = None) -> dict:
    """Four-gate decision for legacy_full / native_full; none for common.

    legacy_full uses the original four gate (median ratio < 1.0, >= 6/8
    groups below 1.0, median cosine >= 0.30, near-TS cosine >= 0.40).
    native_full uses the SAME literal thresholds but is flagged
    ``new_protocol=True`` (a NEW research gate, never a rewrite of G5).
    common_unchanged is not gated — only difference + missing denominators.
    """
    thresholds = dict(DEFAULT_GATE_THRESHOLDS if thresholds is None
                      else thresholds)
    protocol = metrics.get("protocol")
    if protocol not in PROTOCOL_NAMES:
        raise ProtocolError("PROTOCOL_UNKNOWN",
                            f"metrics protocol {protocol!r} not in {PROTOCOL_NAMES}")

    if protocol == "common_unchanged":
        return {
            "protocol": protocol,
            "apply_gate": False,
            "gate_name": None,
            "new_protocol": False,
            "decision": "NOT_GATED",
            "reasons": [
                "common_unchanged is a same-question comparison set; no "
                "four-gate applied — report difference + missing "
                "denominators only (plan §10)"],
            "criteria": {},
            "missing_denominators": metrics.get("missing"),
        }

    new_protocol = protocol == "native_full"
    gate_name = ("native_four_gate_research" if new_protocol
                 else "legacy_four_gate")

    ratio_median = metrics["mse"]["group_equal_weighted"]["ratio_median"]
    ratio_below = metrics["mse"]["group_equal_weighted"][
        "ratio_below_one_groups"]
    cos_median = metrics["direction"]["cosine_median"]
    near_ts_cos_median = metrics["near_ts"]["cosine_median"]

    c1 = math.isfinite(ratio_median) and \
        ratio_median < thresholds["median_mse_ratio_max"]
    c2 = ratio_below >= thresholds["n_groups_ratio_below_one_min"]
    c3 = cos_median is not None and \
        cos_median >= thresholds["median_cosine_min"]
    c4 = near_ts_cos_median is not None and \
        near_ts_cos_median >= thresholds["near_ts_cosine_min"]

    criteria = {
        "median_mse_ratio": {
            "value": ratio_median,
            "threshold": thresholds["median_mse_ratio_max"],
            "pass": bool(c1),
        },
        "n_groups_ratio_below_one": {
            "value": ratio_below,
            "threshold": thresholds["n_groups_ratio_below_one_min"],
            "pass": bool(c2),
        },
        "median_cosine": {
            "value": cos_median,
            "threshold": thresholds["median_cosine_min"],
            "pass": bool(c3),
        },
        "near_ts_cosine": {
            "value": near_ts_cos_median,
            "threshold": thresholds["near_ts_cosine_min"],
            "pass": bool(c4),
        },
    }
    reasons = [
        f"{name}: {c['value']} vs {c['threshold']} -> "
        f"{'pass' if c['pass'] else 'fail'}"
        for name, c in criteria.items()
    ]
    passed = c1 and c2 and c3 and c4
    return {
        "protocol": protocol,
        "apply_gate": True,
        "gate_name": gate_name,
        "new_protocol": new_protocol,
        "new_protocol_note": (
            "native_full reuses the pre-registered 1.0/6-of-8/0.30/0.40 "
            "thresholds as a NEW research gate; passing it never rewrites "
            "historical G5 (contract §7)" if new_protocol else None),
        "decision": "PASS" if passed else "BELOW_GATE",
        "criteria": criteria,
        "reasons": reasons,
    }


__all__ = [
    "ACTION_EPS",
    "COMMON_UNCHANGED_IDENTITY",
    "DEFAULT_GATE_THRESHOLDS",
    "GATE_MEDIAN_COSINE_MIN",
    "GATE_MEDIAN_RATIO_MAX",
    "GATE_NEAR_TS_COSINE_MIN",
    "GATE_N_GROUPS",
    "GATE_N_GROUPS_RATIO_BELOW_ONE_MIN",
    "LEGACY_FULL_IDENTITY",
    "NATIVE_FULL_IDENTITY",
    "NEAR_TS_ARC_FRACTION",
    "NEAR_ZERO_THRESHOLD_ANGSTROM",
    "PROTOCOL_EVAL_SCHEMA",
    "PROTOCOL_FREEZE_SCHEMA",
    "PROTOCOL_NAMES",
    "ProtocolError",
    "apply_gates",
    "evaluate_protocol",
    "freeze_digest",
    "freeze_protocols",
    "load_protocol_reference",
    "verify_freeze_payload",
]
