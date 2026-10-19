#!/usr/bin/env python3
"""V5.1a D3: are the hard clusters HARD (model) or DIRTY (labels)? (plan §3.1).

Builds one diagnostic card per plan-cited hard cluster (plan §3.1 D3:
RXN_1675 / RXN_2309 / RXN_238 / RXN_1484 — resolved to 12-char cluster ids
by numeric suffix against the fit50 split; an absent/ambiguous short id is
RECORDED and the 4 worst OOF clusters by sign accuracy are added, clearly
labeled, per the diagnosis spec).

Each card combines:
* label distributions from the d0=0.05 teacher shards (d_eff / terminal /
  capped / eta / near_ts / reason_id);
* OOF model metrics from the shared five-fold prediction pass (alpha,
  cos, sign accuracy, SRE, RMSE/zero, RMSE/classic);
* branch/family metadata from the split manifest + native sidecars
  (reference gauge tier, gate_ok, tier_gates);
* a nearest-neighbor conflict check (same family AND |d_eff median
  difference| < 0.01 A);
* family-level aggregation (is the WHOLE family degraded?).

Verdict per cluster: ``hard`` / ``suspect-label`` / ``conflicted`` with the
triggering evidence recorded.  Read-only; no training, no network.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _p in (PROJECT_ROOT, PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from diagnose_v51_baseline_oof import (  # noqa: E402
    DEAD_ZONE_MAG,
    D0,
    DEFAULT_CACHE,
    DEFAULT_OUT,
    DEFAULT_RUNS_ROOT,
    DEFAULT_SPLIT,
    EFFECTIVE_ATOM_TAU,
    atom_alphas,
    cluster_metrics,
    collect_oof_records,
    resolve_cluster_suffix,
    sha256_file,
)
from diagnose_v51_baseline_oof import (  # noqa: E402
    _median as median_of,
    _oof_fingerprint as oof_fingerprint,
    _quantile as quantile,
    _utc_now as utc_now,
)
from diagnose_v51_g53_buckets import (  # noqa: E402
    SIGN_THRESHOLD,
    bucketize,
    pair_rows,
)

SCHEMA_VERSION: Final[str] = "pesn_v51_hard_clusters_diag_v1"

#: Plan §3.1 D3 short forms + the spec-cited padded forms (recorded even
#: when they do not resolve — RXN_0000000167 has no fit50 cluster suffix).
DEFAULT_TARGETS: Final[tuple[str, ...]] = (
    "RXN_1675", "RXN_2309", "RXN_238", "RXN_1484",
    "RXN_0000000167",  # spec-cited; expected ABSENT — recorded as finding
)

DEFAULT_TEACHER_ROOT = PROJECT_ROOT / "outputs/pesn/v5/teacher/d0_0.05"
DEFAULT_NATIVE_ROOT = PROJECT_ROOT / "outputs/pesn/v5/native"
DEFAULT_OUT = (PROJECT_ROOT / "outputs/pesn/v5_1/diagnostics"
               / "hard_clusters.json")

#: Verdict thresholds (frozen here, echoed into the output).
DEGENERATE_DEFF_FRACTION: Final[float] = 0.50   # rows with d_eff < 0.005 A
TERMINAL_FRACTION_SUSPECT: Final[float] = 0.60
DEFF_NEIGHBOR_TOL: Final[float] = 0.01          # A, neighbor conflict check


# ---------------------------------------------------------------------------
# label-side extraction from teacher shards
# ---------------------------------------------------------------------------

def teacher_rows_for_cluster(teacher_root: Path, gid: str) -> dict:
    """d_eff/terminal/capped/eta/near_ts/reason distributions for a cluster.

    Reads both branch shards; only ELIGIBLE rows feed label distributions
    (ineligible rows are counted separately by reason_id).
    """
    rows = {"n_rows": 0, "n_eligible": 0, "reason_counts": {},
            "d_eff": [], "eta": [], "near_ts": [],
            "terminal": [], "capped": [], "branches": {}}
    for branch in (0, 1):
        path = teacher_root / f"{gid}.branch{branch}.npz"
        if not path.is_file():
            rows["branches"][str(branch)] = {"status": "missing"}
            continue
        with np.load(path, allow_pickle=False) as d:
            eligible = np.asarray(d["eligible"], dtype=bool)
            reason = np.asarray(d["reason_id"], dtype=np.int64)
            d_eff = np.asarray(d["d_eff"], dtype=np.float64)
            eta = np.asarray(d["eta"], dtype=np.float64)
            near = np.asarray(d["near_ts"], dtype=np.float64)
            terminal = np.asarray(d["terminal"], dtype=bool)
            capped = np.asarray(d["capped"], dtype=bool)
        rows["n_rows"] += int(eligible.size)
        rows["n_eligible"] += int(eligible.sum())
        for rv, c in zip(*np.unique(reason, return_counts=True)):
            rows["reason_counts"][str(int(rv))] = (
                rows["reason_counts"].get(str(int(rv)), 0) + int(c))
        if bool(eligible.any()):
            rows["d_eff"].extend(d_eff[eligible].tolist())
            rows["eta"].extend(eta[eligible].tolist())
            rows["near_ts"].extend(near[eligible].tolist())
            rows["terminal"].extend(terminal[eligible].tolist())
            rows["capped"].extend(capped[eligible].tolist())
        rows["branches"][str(branch)] = {
            "status": "present", "n_rows": int(eligible.size),
            "n_eligible": int(eligible.sum()),
        }
    d_eff = np.asarray(rows["d_eff"], dtype=np.float64)
    terminal = np.asarray(rows["terminal"], dtype=bool)
    capped = np.asarray(rows["capped"], dtype=bool)
    near = np.asarray(rows["near_ts"], dtype=np.float64)
    summary = {
        "n_rows": rows["n_rows"],
        "n_eligible": rows["n_eligible"],
        "reason_counts": rows["reason_counts"],
        "d_eff_median": median_of(d_eff),
        "d_eff_q25": quantile(d_eff, 0.25),
        "d_eff_q75": quantile(d_eff, 0.75),
        "d_eff_min": (float(d_eff.min()) if d_eff.size else None),
        "fraction_d_eff_lt_0.005": (
            float((d_eff < SIGN_THRESHOLD).mean()) if d_eff.size else None),
        "terminal_fraction": (float(terminal.mean()) if terminal.size else None),
        "capped_fraction": (float(capped.mean()) if capped.size else None),
        "near_ts_median": median_of(near),
        "branches": rows["branches"],
    }
    return {"summary": summary, "_d_eff_samples": d_eff.tolist()}


# ---------------------------------------------------------------------------
# native sidecar gauge metadata
# ---------------------------------------------------------------------------

def gauge_metadata(native_root: Path, gid: str) -> dict:
    """Reference-gauge tier / gate evidence from native branch sidecars."""
    out = {}
    for branch in (0, 1):
        path = native_root / f"{gid}.branch{branch}.meta.json"
        if not path.is_file():
            out[str(branch)] = {"status": "missing"}
            continue
        meta = json.loads(path.read_text(encoding="utf-8"))
        ref = dict(meta.get("reference") or {})
        out[str(branch)] = {
            "status": "present",
            "ref_tier": ref.get("tier"),
            "gate_ok": ref.get("gate_ok"),
            "n_rotor_bonds_cut": ref.get("n_rotor_bonds_cut"),
            "key_rows": meta.get("key_rows"),
            "scatter_eigenvalues_angstrom2":
                ref.get("scatter_eigenvalues_angstrom2"),
            "tier_gates": [
                {"tier": t.get("tier"), "gate_ok": t.get("gate_ok"),
                 "n_rows": t.get("n_rows")}
                for t in (ref.get("tier_gates") or [])],
            "frame_layer": meta.get("frame_layer"),
            "n_atoms": meta.get("n_atoms"),
        }
    return out


def uses_gauge_fallback(gauge: dict) -> bool:
    """True when any branch needed a tier BELOW rigid_1_2hop or failed."""
    fallback_tiers = {"any_1_2hop", "any_3hop", "all_nonkey_heavy"}
    for branch in gauge.values():
        if not isinstance(branch, dict):
            continue
        if branch.get("gate_ok") is False:
            return True
        if branch.get("ref_tier") in fallback_tiers:
            return True
    return False


# ---------------------------------------------------------------------------
# family metadata + neighbor conflict
# ---------------------------------------------------------------------------

def family_map(split_path: Path) -> dict[str, str]:
    doc = json.loads(split_path.read_text(encoding="utf-8"))
    out = {}
    for comp in doc.get("components") or []:
        fam = str(comp.get("family", "unknown"))
        for gid in comp.get("group_ids") or []:
            out[str(gid)] = fam
    return out


def neighbor_conflicts(gid: str, fam: str, label_summary: dict,
                       gauge: dict,
                       candidate_cards: dict[str, dict],
                       tol: float = DEFF_NEIGHBOR_TOL) -> list[dict]:
    """Near-duplicate label check: same family + indistinguishable label
    profile.

    The spec's simple rule (same family AND |d_eff median diff| < 0.01 A)
    is applied, but at d0=0.05 EVERY eligible cluster row lands exactly on
    the contact distance (d_eff == d0), so the median component alone is
    non-discriminating — it would flag every co-family pair.  A neighbor
    therefore also needs the same key-atom count (topology), which is the
    remaining near-duplicate signal.  The degeneracy is recorded in each
    card via ``d_eff_median_degenerate``.
    """
    d_eff_median = label_summary.get("d_eff_median")
    n_keys = _gauge_n_keys(gauge)
    if d_eff_median is None:
        return []
    out = []
    for other, info in sorted(candidate_cards.items()):
        if other == gid:
            continue
        other_side = info.get("label_side") or {}
        other_med = other_side.get("d_eff_median")
        if other_med is None:
            continue
        diff = abs(float(other_med) - float(d_eff_median))
        if diff >= tol:
            continue
        other_keys = _gauge_n_keys(info.get("gauge") or {})
        same_topology = (n_keys is not None and other_keys is not None
                         and int(other_keys) == int(n_keys))
        out.append({"cluster": other, "d_eff_median_diff": diff,
                    "same_key_atom_count": bool(same_topology),
                    "near_duplicate_candidate": bool(same_topology)})
    return out


def _gauge_n_keys(gauge: dict) -> int | None:
    for branch in gauge.values():
        if isinstance(branch, dict) and branch.get("key_rows") is not None:
            return len(branch["key_rows"])
    return None


# ---------------------------------------------------------------------------
# card assembly
# ---------------------------------------------------------------------------

def cluster_sign_accuracy(cluster_bucket: dict, gid: str) -> dict:
    """Active-pair (|Δd_true|>=0.005) sign accuracy for one cluster."""
    cb = cluster_bucket.get(gid)
    if not cb:
        return {"sign_accuracy": None, "n_active_pairs": 0}
    n = 0
    correct = 0
    for key in ("1", "2", "3"):
        n += cb[key]["n_pairs"]
        correct += cb[key]["n_sign_correct"]
    return {"sign_accuracy": (correct / n if n else None),
            "n_active_pairs": n}


def build_card(gid: str, *, resolved: dict, teacher_root: Path,
               native_root: Path, families: dict[str, str],
               model_side: dict, cluster_bucket: dict,
               family_agg: dict) -> dict:
    label = teacher_rows_for_cluster(teacher_root, gid)
    gauge = gauge_metadata(native_root, gid)
    sign = cluster_sign_accuracy(cluster_bucket, gid)
    fam = families.get(gid, "unknown")

    # neighbor check: only same-family clusters (avoid re-reading every
    # other family's shards)
    fam_members = sorted(g for g, f in families.items()
                         if f == fam and g != gid)
    candidate_cards = {
        g: {"label_side": teacher_rows_for_cluster(teacher_root, g)["summary"],
            "gauge": gauge_metadata(native_root, g)}
        for g in fam_members}
    all_neighbors = neighbor_conflicts(
        gid, fam, label["summary"], gauge, candidate_cards)
    # only NEAR-DUPLICATES (same family + same key-atom count) count as
    # conflicts; same-family neighbors with different topology are recorded
    # but do not flip the verdict (see neighbor_conflicts docstring)
    conflicts = [c for c in all_neighbors if c["near_duplicate_candidate"]]

    triggers: list[str] = []
    evidence: dict = {}
    notes: list[str] = []
    frac_deg = label["summary"].get("fraction_d_eff_lt_0.005")
    if frac_deg is not None and frac_deg >= DEGENERATE_DEFF_FRACTION:
        triggers.append("near-degenerate d_eff")
        evidence["fraction_d_eff_lt_0.005"] = frac_deg
    term_frac = label["summary"].get("terminal_fraction")
    if term_frac is not None and term_frac >= TERMINAL_FRACTION_SUSPECT:
        triggers.append("high terminal fraction")
        evidence["terminal_fraction"] = term_frac
    gate_failed = any(
        isinstance(v, dict) and v.get("gate_ok") is False
        for v in gauge.values())
    tier_used = any(
        isinstance(v, dict) and v.get("ref_tier") not in (None, "rigid_1_2hop")
        for v in gauge.values())
    if gate_failed:
        # a failed gate means a branch fell through to gauge_unresolved ->
        # its rows are INELIGIBLE: the cluster loses training labels there
        triggers.append("gauge gate failed on a branch")
        evidence["gauge_gate_failed"] = {
            b: {"ref_tier": v.get("ref_tier"), "gate_ok": v.get("gate_ok")}
            for b, v in gauge.items()
            if isinstance(v, dict) and v.get("gate_ok") is False}
    if tier_used:
        # NOTE (not a trigger): 88/100 fit50 branches resolve on a
        # non-rigid tier, so tier use alone is not discriminative
        notes.append("reference resolved on a non-rigid tier "
                     "(common: 88/100 fit50 branches; not a label-quality "
                     "signal by itself)")
        evidence["gauge_tier"] = {
            b: v.get("ref_tier") for b, v in gauge.items()
            if isinstance(v, dict)}
    if conflicts:
        triggers.append("near-neighbor label conflict")
        evidence["conflicts"] = conflicts
    if all_neighbors:
        evidence["same_family_neighbors"] = all_neighbors
    evidence["d_eff_median_degenerate"] = (
        label["summary"].get("d_eff_median") is not None
        and abs(float(label["summary"]["d_eff_median"]) - D0) < 1e-6)
    if evidence["d_eff_median_degenerate"]:
        notes.append("d_eff median == d0 for this cluster; at d0=0.05 ALL "
                     "49 fit50 clusters have d_eff median == d0, so the "
                     "|Δd_eff median| < 0.01 conflict rule is "
                     "non-discriminating here — near-duplicate status "
                     "additionally requires matching key-atom count")

    if conflicts:
        verdict = "conflicted"
    elif triggers:
        verdict = "suspect-label"
    else:
        verdict = "hard"

    fam_entry = family_agg.get(fam)
    fam_note = None
    if fam_entry and fam_entry["n_clusters"] > 1:
        fam_accs = [v for v in fam_entry["sign_accuracies"] if v is not None]
        degraded = [a for a in fam_accs if a < 0.75]
        fam_note = {
            "family": fam,
            "n_clusters": fam_entry["n_clusters"],
            "family_sign_accuracy_median": (
                float(np.median(fam_accs)) if fam_accs else None),
            "n_clusters_sign_below_0.75": len(degraded),
            "whole_family_degraded": (
                bool(fam_accs) and len(degraded) == len(fam_accs)),
        }

    return {
        "cluster": gid,
        "resolution": resolved,
        "family": fam,
        "verdict": verdict,
        "verdict_triggers": triggers,
        "verdict_notes": notes,
        "verdict_evidence": evidence,
        "label_side": label["summary"],
        "gauge_sidecar": gauge,
        "uses_gauge_fallback": uses_gauge_fallback(gauge),
        "model_side": model_side,
        "sign": sign,
        "family_level": fam_note,
    }


def model_metrics_for(records: Sequence[dict], gid: str) -> dict:
    sub = [r for r in records if r["group_id"] == gid]
    if not sub:
        return {"n_samples": 0}
    per_cluster, _agg = cluster_metrics(sub)
    entry = dict(per_cluster.get(gid, {}))
    alphas = np.zeros(0)
    dead_num = 0
    dead_den = 0
    cos_all: list[float] = []
    for r in sub:
        a, eff = atom_alphas(r["u"], r["y"])
        alphas = np.concatenate([alphas, a])
        n_eff = int(eff.sum())
        if n_eff:
            u_norm = np.linalg.norm(r["u"][eff], axis=-1)
            dead_num += int((u_norm < DEAD_ZONE_MAG).sum())
            dead_den += n_eff
    entry.update({
        "n_samples": len(sub),
        "alpha_median": median_of(alphas),
        "alpha_q25": quantile(alphas, 0.25),
        "alpha_q75": quantile(alphas, 0.75),
        "dead_zone_fraction": (float(dead_num) / dead_den if dead_den else None),
        "n_effective_atoms": int(alphas.size),
    })
    _ = cos_all
    return entry


def worst_clusters_by_sign(cluster_bucket: dict, n: int = 4) -> list[dict]:
    ranked = []
    for gid in sorted(cluster_bucket):
        s = cluster_sign_accuracy(cluster_bucket, gid)
        if s["n_active_pairs"] > 0 and s["sign_accuracy"] is not None:
            ranked.append({"cluster": gid, **s})
    ranked.sort(key=lambda d: (d["sign_accuracy"], d["cluster"]))
    return ranked[:n]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--runs-root", default=str(DEFAULT_RUNS_ROOT))
    parser.add_argument("--split-manifest", default=str(DEFAULT_SPLIT))
    parser.add_argument("--teacher-root", default=str(DEFAULT_TEACHER_ROOT))
    parser.add_argument("--native-root", default=str(DEFAULT_NATIVE_ROOT))
    parser.add_argument("--targets", nargs="*", default=None,
                        help="short/long cluster ids (default: plan §3.1 D3)")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    runs_root = Path(args.runs_root)
    split_path = Path(args.split_manifest)
    teacher_root = Path(args.teacher_root)
    native_root = Path(args.native_root)
    fingerprint = oof_fingerprint(runs_root, split_path, D0)
    fingerprint["teacher_shard"] = sha256_file(
        teacher_root / "RXN_0000000072.branch0.npz")

    families = family_map(split_path)
    known = sorted(families)
    targets = list(args.targets) if args.targets else list(DEFAULT_TARGETS)

    resolution = [resolve_cluster_suffix(t, known) for t in targets]
    resolved_ids = []
    findings = []
    for res in resolution:
        if res["status"] == "resolved":
            gid = res["matches"][0]
            if gid not in resolved_ids:
                resolved_ids.append(gid)
        else:
            findings.append(
                f"target {res['requested']!r} is {res['status']} in fit50 "
                f"(matches={res['matches']}); recorded per D3 spec")

    records = collect_oof_records(
        runs_root=runs_root, split_path=split_path, d0=D0,
        device_name=args.device, batch_size=args.batch_size,
        cache_path=(None if args.no_cache else Path(args.cache)),
        use_cache=not args.no_cache)
    rows = pair_rows(records)
    buckets = bucketize(rows)
    cluster_bucket = buckets["per_cluster_bucket"]

    fallback_used = False
    fallback_note = None
    if findings:
        fallback_used = True
        worst = worst_clusters_by_sign(cluster_bucket, n=4)
        fallback_note = (
            "one or more plan-cited short ids did not resolve; added the 4 "
            "worst OOF clusters by active-pair sign accuracy, labeled "
            "'fallback_worst_sign_accuracy' (D3 spec)")
        for w in worst:
            if w["cluster"] not in resolved_ids:
                resolved_ids.append(w["cluster"])

    # family-level aggregation over ALL 49 OOF clusters
    family_agg: dict[str, dict] = {}
    for gid in sorted(cluster_bucket):
        fam = families.get(gid, "unknown")
        entry = family_agg.setdefault(
            fam, {"n_clusters": 0, "sign_accuracies": [], "clusters": []})
        entry["n_clusters"] += 1
        entry["clusters"].append(gid)
        entry["sign_accuracies"].append(
            cluster_sign_accuracy(cluster_bucket, gid)["sign_accuracy"])

    spec_ids = {res["matches"][0] for res in resolution
                if res["status"] == "resolved"}
    worst_set = {w["cluster"] for w in worst_clusters_by_sign(cluster_bucket)}
    cards = []
    for gid in resolved_ids:
        resolved_meta = next(
            (r for r in resolution
             if r["status"] == "resolved" and r["matches"][0] == gid), None)
        label_kind = ("plan_cited" if gid in spec_ids
                      else "fallback_worst_sign_accuracy")
        resolved = {
            "match_rule": (resolved_meta or {}).get("rule"),
            "requested": (resolved_meta or {}).get("requested"),
            "label_kind": label_kind,
        }
        cards.append(build_card(
            gid, resolved=resolved, teacher_root=teacher_root,
            native_root=native_root, families=families,
            model_side=model_metrics_for(records, gid),
            cluster_bucket=cluster_bucket, family_agg=family_agg))
    _ = (fallback_used, worst_set)

    doc = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "d0": D0,
        "targets_requested": targets,
        "resolution": resolution,
        "findings": findings,
        "fallback": {"used": fallback_used, "note": fallback_note,
                     "worst_clusters_by_sign": worst_clusters_by_sign(
                         cluster_bucket)},
        "thresholds": {
            "degenerate_d_eff_fraction_min": DEGENERATE_DEFF_FRACTION,
            "terminal_fraction_suspect_min": TERMINAL_FRACTION_SUSPECT,
            "neighbor_d_eff_median_tol_angstrom": DEFF_NEIGHBOR_TOL,
            "sign_pair_threshold_angstrom": SIGN_THRESHOLD,
            "effective_atom_tau_angstrom": EFFECTIVE_ATOM_TAU,
            "verdict_precedence": "conflicted > suspect-label > hard",
        },
        "cards": cards,
        "fingerprint": fingerprint,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"[hard_clusters] wrote {out}  ({len(cards)} cards)")
    for card in cards:
        print(f"  {card['cluster']} [{card['resolution']['label_kind']}] "
              f"-> {card['verdict']} "
              f"(triggers: {', '.join(card['verdict_triggers']) or '-'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
