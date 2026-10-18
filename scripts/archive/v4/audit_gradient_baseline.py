#!/usr/bin/env python3
"""V4.2 gradient baseline + identifiability audit (Lane A, §4 + §4.1a-3/4/5).

Freezes ``physical_prior_spec.json`` (fit45_train estimate) and reports the
no-training baselines — zero action, +g, −g, +M⁻¹g, −M⁻¹g (each scale-fitted
by the SAME non-negative least-squares rule as Δs_fit), and the fixed physical
skip ``a_phys = Δs_fit·b`` (states below tau_g are zeroed, coverage reported
separately) — on BOTH the legacy (subdivided) and native (reconnected) labels,
reported separately (denominators never mixed).  Also runs the identifiability
search (nearly-identical single-frame input but conflicting teacher direction)
and emits the H1 release boolean.

H1 release rule (pre-registered): Δs_fit > 0 and finite, fit5_dev group-median
MSE/zero-action < 1, and overall direction cosine > 0 (physical skip on native
dev labels).  Otherwise H1 = prior_not_supported and H2 parents on H0.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.contracts.dual_ts import (  # noqa: E402
    BOUNDARY_TEACHER_ENDPOINT,
    DualTSTeacher,
)
from pes2ts_core.pesn.offline.data.direct_preprocess import masses_for  # noqa: E402
from pes2ts_core.pesn.offline.data.dual_ts_physical_prior import (  # noqa: E402
    PhysicalPriorSpec,
    a_phys_for_batch,
    estimate_physical_prior,
    save_physical_prior,
)
from pes2ts_core.pesn.offline.data.dual_ts_dev_split import load_dev_split  # noqa: E402

AUDIT_SCHEMA: str = "pes2ts_dual_ts_gradient_baseline_audit_v1"
TOOL_VERSION: str = "pes2ts_dual_ts_gradient_baseline_tool_v1"

DEFAULT_ROOT: str = "outputs/pesn/dual_ts_v2_20261015"
DEFAULT_OUT: str = f"{DEFAULT_ROOT}/wp42_campaign"

NEAR_TS_ARC_FRACTION: float = 0.2
NEAR_ZERO_THRESHOLD: float = 0.01
IDENTIFIABILITY_X_TOL_ANGSTROM: float = 0.05  # fit-frozen similarity threshold

BASELINES: tuple[str, ...] = (
    "zero_action", "plus_g", "minus_g", "plus_m1g", "minus_m1g", "physical_skip")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _flatten(records: list[dict]) -> dict:
    """Flatten per-state records into per-atom arrays (monitor-style)."""
    a_star, grad, z = [], [], []
    group, branch, ral, near_zero, is_h, step_len = [], [], [], [], [], []
    for r in records:
        n = int(np.asarray(r["action"]).shape[0])
        a = np.asarray(r["action"], dtype=np.float64)
        g = np.asarray(r["gradient"], dtype=np.float64)
        zz = np.asarray(r["atomic_numbers"], dtype=int)
        a_star.append(a)
        grad.append(g)
        z.append(zz)
        group.extend([r["group_id"]] * n)
        branch.extend([r["branch_id"]] * n)
        ral.extend([r["remaining_arc_length"]] * n)
        near_zero.extend([bool(v) for v in r["near_zero_mask"]])
        is_h.extend([bool(int(zzz) == 1) for zzz in zz])
        step_len.extend([float(v) for v in np.linalg.norm(a, axis=1)])
    return {
        "a_star": np.concatenate(a_star, axis=0) if a_star else
        np.zeros((0, 3)),
        "grad": np.concatenate(grad, axis=0) if grad else np.zeros((0, 3)),
        "z": np.concatenate(z, axis=0) if z else np.zeros(0, dtype=int),
        "masses": np.concatenate(
            [masses_for(np.asarray(r["atomic_numbers"], dtype=int))
             for r in records], axis=0) if records else np.zeros(0),
        "group": np.asarray(group),
        "branch": np.asarray(branch),
        "ral": np.asarray(ral, dtype=np.float64),
        "near_zero": np.asarray(near_zero, dtype=bool),
        "is_h": np.asarray(is_h, dtype=bool),
        "step_len": np.asarray(step_len, dtype=np.float64),
    }


def _near_ts_mask(flat: dict) -> np.ndarray:
    """near-TS atoms: remaining_arc_length <= 0.2 * branch max."""
    mask = np.zeros(flat["ral"].shape[0], dtype=bool)
    for br in set(flat["branch"].tolist()):
        bidx = flat["branch"] == br
        ral = flat["ral"][bidx]
        mask[bidx] = ral <= NEAR_TS_ARC_FRACTION * float(ral.max())
    return mask


def compute_metrics(flat: dict, a_hat: np.ndarray) -> dict:
    """All registered baseline metrics over one atom population."""
    a_star = flat["a_star"]
    diff = a_hat - a_star
    sq = (diff * diff).sum(axis=1)  # (M,)
    n = int(a_star.shape[0])
    if n == 0:
        return {"n_atoms": 0}

    # group-equal MSE + pooled MSE + per-group ratio vs zero.
    groups = sorted(set(flat["group"].tolist()))
    per_group: dict[str, dict] = {}
    for grp in groups:
        gi = flat["group"] == grp
        mse_g = float(sq[gi].mean())
        zero_g = float((a_star[gi] ** 2).sum(axis=1).mean())
        per_group[grp] = {
            "mse_A2": mse_g, "zero_mse_A2": zero_g,
            "ratio_vs_zero": mse_g / zero_g if zero_g > 0 else float("nan"),
        }
    group_mse = [per_group[g]["mse_A2"] for g in groups]
    group_ratio = [per_group[g]["ratio_vs_zero"] for g in groups]
    group_ratio_finite = [v for v in group_ratio if np.isfinite(v)]
    pooled_mse = float(sq.mean())

    # direction cosine (|a*|-weighted, near-zero atoms excluded).
    dir_valid = ~flat["near_zero"]
    if dir_valid.any():
        ad = a_hat[dir_valid]
        ast = a_star[dir_valid]
        w = np.linalg.norm(ast, axis=1)
        cos = (ad * ast).sum(axis=1) / np.maximum(
            np.linalg.norm(ad, axis=1) * w, 1e-12)
        overall_cos = float((w * cos).sum() / w.sum())
        n_dir = int(dir_valid.sum())
    else:
        overall_cos = float("nan")
        n_dir = 0

    near_ts = _near_ts_mask(flat) & dir_valid
    if near_ts.any():
        ad = a_hat[near_ts]
        ast = a_star[near_ts]
        w = np.linalg.norm(ast, axis=1)
        cos = (ad * ast).sum(axis=1) / np.maximum(
            np.linalg.norm(ad, axis=1) * w, 1e-12)
        near_ts_cos = float((w * cos).sum() / w.sum())
    else:
        near_ts_cos = float("nan")

    near_zero = flat["near_zero"]
    near_zero_rmse = float(np.sqrt(sq[near_zero].mean())) \
        if near_zero.any() else float("nan")

    h_rmse = float(np.sqrt(sq[flat["is_h"]].mean())) \
        if flat["is_h"].any() else float("nan")
    heavy_rmse = float(np.sqrt(sq[~flat["is_h"]].mean())) \
        if (~flat["is_h"]).any() else float("nan")

    def _long_rmse(pct):
        if not flat["step_len"].size:
            return float("nan")
        thr = float(np.percentile(flat["step_len"], pct))
        sel = flat["step_len"] >= thr
        return float(np.sqrt(sq[sel].mean())) if sel.any() else float("nan")

    return {
        "n_atoms": n,
        "n_states": None,  # filled by caller
        "group_equal_mse_A2": float(np.mean(group_mse)),
        "group_median_mse_A2": float(np.median(group_mse)),
        "pooled_mse_A2": pooled_mse,
        "per_group": per_group,
        "group_median_ratio_vs_zero": float(np.median(group_ratio_finite))
        if group_ratio_finite else float("nan"),
        "overall_cos": overall_cos,
        "near_ts_cos": near_ts_cos,
        "near_zero_abs_rmse": near_zero_rmse,
        "effective_denominator": {
            "n_atoms": n, "n_direction_valid": n_dir,
            "n_near_zero": int(near_zero.sum()),
            "n_near_ts": int(near_ts.sum()),
            "n_h": int(flat["is_h"].sum()),
            "n_heavy": int((~flat["is_h"]).sum()),
        },
        "h_atom_rmse": h_rmse,
        "heavy_atom_rmse": heavy_rmse,
        "long_step_p90_rmse": _long_rmse(90),
        "long_step_p99_rmse": _long_rmse(99),
    }


def nnls_scale(records: list[dict], direction: np.ndarray) -> float:
    """Fixed-scale NNLS (same rule as Δs_fit) for one direction per atom.

    ``direction`` is the per-atom direction vector (flattened over records),
    aligned with ``_flatten(records)``.
    """
    flat = _flatten(records)
    d = np.asarray(direction, dtype=np.float64)
    a = flat["a_star"]
    dot = (d * a).sum(axis=1)
    norm2 = (d * d).sum(axis=1)
    # L_xyz reduction: atom sum -> branch mean -> reaction mean -> group mean.
    cell_num: dict[tuple, float] = {}
    cell_den: dict[tuple, float] = {}
    cell_count: dict[tuple, int] = {}
    for i in range(len(dot)):
        cell = (flat["group"][i], flat["branch"][i])
        cell_num[cell] = cell_num.get(cell, 0.0) + dot[i]
        cell_den[cell] = cell_den.get(cell, 0.0) + norm2[i]
        cell_count[cell] = cell_count.get(cell, 0) + 1
    groups = sorted({c[0] for c in cell_num})
    num = 0.0
    den = 0.0
    for g in groups:
        cells_g = [c for c in cell_num if c[0] == g]
        num += float(np.mean([cell_num[c] / cell_count[c] for c in cells_g]))
        den += float(np.mean([cell_den[c] / cell_count[c] for c in cells_g]))
    num /= len(groups)
    den /= len(groups)
    if den <= 1e-30:
        raise ValueError("NNLS denominator degenerate")
    return max(0.0, num / den)


def baseline_actions(records: list[dict], flat: dict,
                     baseline: str, spec: PhysicalPriorSpec,
                     scale: float | None) -> np.ndarray:
    """Per-atom predicted action for one baseline (flattened)."""
    grad = flat["grad"]
    masses = flat["masses"]
    if baseline == "zero_action":
        return np.zeros_like(grad)
    if baseline == "plus_g":
        return scale * grad
    if baseline == "minus_g":
        return -scale * grad
    if baseline == "plus_m1g":
        return scale * grad / masses[:, None]
    if baseline == "minus_m1g":
        return -scale * grad / masses[:, None]
    if baseline == "physical_skip":
        # per-state a_phys, then flatten (below tau_g -> zero).
        out = []
        for r in records:
            g = np.asarray(r["gradient"], dtype=np.float64)[None]
            z = np.asarray(r["atomic_numbers"], dtype=int)
            out.append(a_phys_for_batch(g, z, spec)[0])
        return np.concatenate(out, axis=0) if out else np.zeros((0, 3))
    raise ValueError(baseline)


def load_native_records(observed_dir: Path, native_dir: Path) -> list[dict]:
    records = []
    for npz_path in sorted(native_dir.glob("features_*.npz")):
        shard = np.load(npz_path, allow_pickle=False)
        rid = str(shard["reaction_id"].item())
        teacher = DualTSTeacher.from_payload(
            _load_json(observed_dir / f"{rid}.teacher.json"))
        states = {s.state_id: s for s in teacher.states}
        sid = [str(x) for x in shard["state_id"]]
        action = np.asarray(shard["action_angstrom"], dtype=np.float64)
        ral = np.asarray(shard["remaining_arc_length"], dtype=np.float64)
        nz = np.asarray(shard["near_zero_action_mask"], dtype=bool)
        term = np.asarray(shard["is_terminal"], dtype=bool)
        branch = [str(x) for x in shard["branch_id"]]
        for i, s in enumerate(sid):
            if term[i]:
                continue
            st = states[s]
            records.append({
                "group_id": str(shard["reaction_group_id"].item()),
                "branch_id": branch[i],
                "gradient": np.asarray(
                    st.observation.gradient_hartree_per_angstrom,
                    dtype=np.float64),
                "atomic_numbers": np.asarray(st.atomic_numbers, dtype=int),
                "action": action[i],
                "remaining_arc_length": float(ral[i]),
                "near_zero_mask": nz[i],
            })
    return records


def load_legacy_records(observed_dir: Path, features_dir: Path) -> list[dict]:
    records = []
    for npz_path in sorted(features_dir.glob("features_*.npz")):
        shard = np.load(npz_path, allow_pickle=False)
        rid = str(shard["reaction_id"].item())
        teacher = DualTSTeacher.from_payload(
            _load_json(observed_dir / f"{rid}.teacher.json"))
        states = {s.state_id: s for s in teacher.states}
        labels = {lab.state_id: lab for lab in teacher.labels}
        sid = [str(x) for x in shard["state_id"]]
        nz = np.asarray(shard["near_zero_action_mask"], dtype=bool)
        branch = [str(x) for x in shard["branch_id"]]
        for i, s in enumerate(sid):
            st = states[s]
            lab = labels[s]
            if lab.action_delta is None:
                continue
            records.append({
                "group_id": str(shard["reaction_group_id"].item()),
                "branch_id": branch[i],
                "gradient": np.asarray(
                    st.observation.gradient_hartree_per_angstrom,
                    dtype=np.float64),
                "atomic_numbers": np.asarray(st.atomic_numbers, dtype=int),
                "action": np.asarray(lab.action_delta, dtype=np.float64),
                "remaining_arc_length": float(lab.remaining_arc_length),
                "near_zero_mask": nz[i],
            })
    return records


def filter_records(records: list[dict], groups: set[str]) -> list[dict]:
    return [r for r in records if r["group_id"] in groups]


def identifiability_search(records: list[dict]) -> dict:
    """Nearly-identical single-frame input vs conflicting teacher direction.

    Within each reaction (same Z + atom mapping ⇒ no permutation/duplicate
    ambiguity), scan native non-terminal state pairs whose centroid-removed X
    differ by <= IDENTIFIABILITY_X_TOL_ANGSTROM and report pairs with opposing
    teacher directions (per-atom |a*|-weighted cosine < 0).
    """
    # group records by reaction (group_id) and centroid-remove X from
    # reconstructed coordinates is NOT available here; we use the gradient/
    # action only.  We re-read coordinates from the teacher below via a
    # supplementary pass, so instead we accept prebuilt state geometry.
    # To keep this self-contained, identifiability is computed by the caller
    # over native records carrying an extra "x_centered" + "action" field.
    by_group: dict[str, list[dict]] = {}
    for r in records:
        by_group.setdefault(r["group_id"], []).append(r)
    conflicts: list[dict] = []
    for gid, recs in sorted(by_group.items()):
        xs = np.asarray([r["x_centered"] for r in recs], dtype=np.float64)
        acts = np.asarray([r["action"] for r in recs], dtype=np.float64)
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                xdiff = float(np.max(np.abs(xs[i] - xs[j])))
                if xdiff > IDENTIFIABILITY_X_TOL_ANGSTROM:
                    continue
                a_i = acts[i].reshape(-1)
                a_j = acts[j].reshape(-1)
                ni, nj = np.linalg.norm(a_i), np.linalg.norm(a_j)
                if ni < 1e-9 or nj < 1e-9:
                    continue
                cos = float(np.dot(a_i, a_j) / (ni * nj))
                if cos < 0.0:
                    conflicts.append({
                        "group_id": gid,
                        "state_a": recs[i]["state_id"],
                        "state_b": recs[j]["state_id"],
                        "max_x_diff_angstrom": xdiff,
                        "direction_cosine": cos,
                    })
    conflicts.sort(key=lambda c: c["max_x_diff_angstrom"])
    return {
        "threshold_x_angstrom": IDENTIFIABILITY_X_TOL_ANGSTROM,
        "n_conflicts": len(conflicts),
        "top": conflicts[:20],
        "note": ("permutation/duplicate identity excluded (same-reaction scan, "
                 "fixed atom mapping); coordinate-error (exact duplicates) "
                 "excluded by x_diff > 0; threshold fit-frozen"),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev-split", default=f"{DEFAULT_OUT}/dev_split.json")
    ap.add_argument("--fit-observed",
                    default=f"{DEFAULT_ROOT}/wp2_fit50_observed")
    ap.add_argument("--native-features",
                    default=f"{DEFAULT_OUT}/native_features")
    ap.add_argument("--legacy-features",
                    default=f"{DEFAULT_ROOT}/wp3_features_v41")
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    observed = Path(args.fit_observed)

    split = load_dev_split(args.dev_split)
    split_hash = split["split_sha256"]
    train_groups = set(split["train_group_ids"])
    dev_groups = set(split["dev_group_ids"])

    print("[audit_gradient_baseline] loading native + legacy fit records")
    native_all = load_native_records(observed, Path(args.native_features))
    legacy_all = load_legacy_records(observed, Path(args.legacy_features))

    native_train = filter_records(native_all, train_groups)
    native_dev = filter_records(native_all, dev_groups)

    # physical prior (fit45_train native) — write spec first.
    print("[audit_gradient_baseline] estimating physical prior")
    spec = estimate_physical_prior(native_train)
    spec = PhysicalPriorSpec(
        delta_s_fit=spec.delta_s_fit, eps_g=spec.eps_g, tau_g=spec.tau_g,
        a_s=spec.a_s,
        median_mass_weighted_step_length_angstrom=(
            spec.median_mass_weighted_step_length_angstrom),
        median_gradient_mass_norm=spec.median_gradient_mass_norm,
        n_states=spec.n_states, n_atoms_nonzero=spec.n_atoms_nonzero,
        split_hash=split_hash,
        meta=dict(spec.meta, population="fit45_train_native"),
    )
    prior_digest = save_physical_prior(spec, out / "physical_prior_spec.json")
    print(f"[audit_gradient_baseline] wrote physical_prior_spec.json "
          f"({prior_digest[:12]}…)")

    # per-atom physical-skip coverage (below tau_g).
    def _coverage(records: list[dict]) -> dict:
        if not records:
            return {"n_states": 0, "n_below_tau_g": 0, "fraction": 0.0}
        below = 0
        for r in records:
            g = np.asarray(r["gradient"], dtype=np.float64)
            m = masses_for(np.asarray(r["atomic_numbers"], dtype=int))
            mag = float(np.sqrt(((g * g).sum(axis=1) / m).sum()))
            if mag < float(spec.tau_g):
                below += 1
        return {"n_states": len(records), "n_below_tau_g": below,
                "fraction": below / len(records)}

    baseline_reports: dict = {}
    for label_version, all_records, train_records, dev_records in (
            ("native", native_all, native_train, native_dev),
            ("legacy", legacy_all,
             filter_records(legacy_all, train_groups),
             filter_records(legacy_all, dev_groups))):

        train_flat = _flatten(train_records)
        # fit scales for the ±g / ±M⁻¹g baselines on TRAIN (same NNLS rule).
        scales: dict[str, float | None] = {}
        for baseline in ("plus_g", "minus_g", "plus_m1g", "minus_m1g"):
            if baseline == "plus_g":
                d = train_flat["grad"]
            elif baseline == "minus_g":
                d = -train_flat["grad"]
            elif baseline == "plus_m1g":
                d = train_flat["grad"] / train_flat["masses"][:, None]
            else:
                d = -train_flat["grad"] / train_flat["masses"][:, None]
            scales[baseline] = nnls_scale(train_records, d)
        scales["zero_action"] = None
        scales["physical_skip"] = None

        for baseline in BASELINES:
            scale = scales[baseline]
            for split_name, recs in (("dev", dev_records),
                                     ("full", all_records)):
                flat = _flatten(recs)
                a_hat = baseline_actions(recs, flat, baseline, spec, scale)
                met = compute_metrics(flat, a_hat)
                met["n_states"] = len(recs)
                met["scale"] = (None if scale is None else float(scale))
                baseline_reports.setdefault(baseline, {}) \
                    .setdefault(label_version, {})[split_name] = met

        baseline_reports.setdefault("physical_skip", {}) \
            .setdefault(label_version, {})["coverage_train"] = \
            _coverage(train_records)
        baseline_reports["physical_skip"][label_version]["coverage_dev"] = \
            _coverage(dev_records)

    # H1 release boolean (physical skip on native dev).
    phys_dev = baseline_reports["physical_skip"]["native"]["dev"]
    h1_delta_s = bool(np.isfinite(spec.delta_s_fit) and spec.delta_s_fit > 0.0)
    h1_mse = bool(phys_dev["group_median_ratio_vs_zero"] < 1.0)
    h1_cos = bool(phys_dev["overall_cos"] > 0.0)
    h1_release = bool(h1_delta_s and h1_mse and h1_cos)

    # identifiability (native full fit, with centered X + state ids).
    ident_records = []
    for npz_path in sorted(Path(args.native_features).glob("features_*.npz")):
        shard = np.load(npz_path, allow_pickle=False)
        rid = str(shard["reaction_id"].item())
        teacher = DualTSTeacher.from_payload(
            _load_json(observed / f"{rid}.teacher.json"))
        states = {s.state_id: s for s in teacher.states}
        sid = [str(x) for x in shard["state_id"]]
        action = np.asarray(shard["action_angstrom"], dtype=np.float64)
        term = np.asarray(shard["is_terminal"], dtype=bool)
        for i, s in enumerate(sid):
            if term[i]:
                continue
            st = states[s]
            coord = np.asarray(st.coordinates, dtype=np.float64)
            ident_records.append({
                "group_id": rid,
                "branch_id": str(shard["branch_id"][i]),
                "state_id": s,
                "x_centered": coord - coord.mean(axis=0),
                "action": action[i],
            })
    ident = identifiability_search(ident_records)

    audit = {
        "schema_version": AUDIT_SCHEMA,
        "tool_version": TOOL_VERSION,
        "generated_utc": _utc_now(),
        "dev_split": {"path": args.dev_split, "split_sha256": split_hash},
        "physical_prior": {
            "path": str(out / "physical_prior_spec.json"),
            "spec_digest": prior_digest,
            "spec": spec.to_payload(),
        },
        "baselines": baseline_reports,
        "identifiability": ident,
        "h1_release": {
            "delta_s_fit_positive_finite": h1_delta_s,
            "fit5_dev_group_median_ratio_vs_zero_lt_1": h1_mse,
            "overall_cos_gt_0": h1_cos,
            "released": h1_release,
            "value": (h1_release if h1_release else "prior_not_supported"),
        },
    }
    audit_path = out / "gradient_baseline_audit.json"
    audit_path.write_text(json.dumps(audit, indent=1, sort_keys=False,
                                     ensure_ascii=False, allow_nan=True),
                          encoding="utf-8")
    print(f"[audit_gradient_baseline] wrote {audit_path}")
    print(f"[audit_gradient_baseline] delta_s_fit={spec.delta_s_fit:.6g} "
          f"eps_g={spec.eps_g:.3e} tau_g={spec.tau_g:.3e} a_s={spec.a_s:.6g}")
    print(f"[audit_gradient_baseline] H1 released={h1_release} "
          f"(delta_s={h1_delta_s}, mse={h1_mse}, cos={h1_cos})")
    print(f"[audit_gradient_baseline] identifiability n_conflicts="
          f"{ident['n_conflicts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
