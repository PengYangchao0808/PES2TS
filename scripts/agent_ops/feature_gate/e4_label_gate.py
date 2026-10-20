#!/usr/bin/env python3
"""G3a label-gate report generator for the V5.2 q/TSStop labels (E4, W-FEAT).

Builds the E4 label report from the REAL E1 six tables (or, with
``--fixture``, from the deterministic ``v3_fixtures`` six-table writer) and
writes it as JSON.  Evidence covered (30h plan §5.2–§5.6 + G3a):

* per-fold pos/neg/masked stop-label coverage (full denominator AND
  train-eligible ``label_status=ok`` rows) plus the declared engineering
  ``insufficient_stop_positives`` precheck (NEVER fabricates positives —
  an insufficient fold makes the Stop arm exploratory-only, plan §5.3);
* q distribution stats per fold, zero-arc mask list, A/TS/B anchor values;
* reverse-intent integrity: ``q(X, Reverse(G)) == −q(X, G)`` exactly
  (frozen yaml ``qlabel.reverse q=-q``);
* frozen stop class weights estimated from INNER-TRAIN rows ONLY
  (dev folds ``{0,1,2}``: fold f's weights come from outer-train minus f;
  confirm weights from all dev folds) — T3.3 provenance, frozen here for
  E5 to consume verbatim;
* head wiring: exact param counts (backbone 55,441 reported SEPARATELY
  from the 2,178 head params), output shapes, and the bitwise
  backbone-unchanged guarantee on a shared R3-style backbone;
* the G3a **Move non-degradation hook** — a DESIGN section only (R3
  checkpoints are pending; status ``not_run``, never faked).

Labels are derived ONLY from the six-table teacher columns (no ground-truth
reads): irc_curve ``ts_index``/endpoints + irc_node geometry/``atomic_numbers``
+ sample_view identity + reaction_intent direction.  Usage::

    python3 scripts/agent_ops/feature_gate/e4_label_gate.py \\
        --tables-dir outputs/pesn/30h_campaign/data \\
        --report outputs/pesn/30h_campaign/e4_labels/e4_label_report.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    STOP_LABEL_SCHEMA_V52,
    ZERO_ARC_EPS_ANGSTROM_V52,
    KeyAtomDatasetV3,
    V52LabelTable,
)

REPORT_SCHEMA: str = "pes2ts_e4_label_report_v1"

#: ENGINEERING precheck for ``insufficient_stop_positives`` (plan §5.3 has
#: no frozen numeric threshold; these declared constants are reported
#: verbatim and may be re-ruled by the supervisor — never silently tuned).
INSUFFICIENT_MIN_POS: int = 20
INSUFFICIENT_MIN_POS_CURVES: int = 5

#: Dev vs confirm folds (campaign ``v52_training.run_matrix``).
DEV_FOLDS: tuple[int, ...] = (0, 1, 2)
CONFIRM_FOLDS: tuple[int, ...] = (3, 4)

Q_DEFINITION: str = (
    "mass-weighted arc-length segmentation along the canonical A->B IRC "
    "path (ds_i = sqrt(sum_a m_a*||dx_a||^2 / sum_a m_a), amu table "
    "ATOMIC_MASS_TABLE_V52, no inter-frame Kabsch: single verified frame "
    "per curve, PD-9); q = -(s_TS-s_i)/(s_TS-s_A) for i<=TS, "
    "(s_i-s_TS)/(s_B-s_TS) for i>TS; A=-1, TS=0, B=+1; reverse intent "
    "q=-q; zero-arc segments masked (eps=%g A)" % ZERO_ARC_EPS_ANGSTROM_V52)
STOP_DEFINITION: str = (
    "d_key_TS = Kabsch-aligned (proper rotation, det=+1, no reflection) "
    "RMSD over the intent K atoms between the current frame and the "
    "trusted TS frame (irc_curve.ts_index); stop=1 if d<=0.04 A, "
    "stop=0 if d>=0.12 A, otherwise Mask (-1); windows inclusive, "
    "NO widening (frozen yaml v52_training.stop_labels)")
MOVE_NON_DEGRADATION_DESIGN: dict = {
    "status": "not_run",
    "reason": "R3 G2 checkpoints pending — comparison executes in G3a/G3 "
              "once R3 baseline run_records exist (H12 rule: no R3 "
              "baseline => V5.2 contract/label tests only)",
    "question": "Does the V5.2 JOINT model (Move+Progress+TSStop, arm QT) "
                "degrade the Move head vs the matched R3-only baseline?",
    "protocol": {
        "pairs": "same (fold, seed, curve set, batch composition, update "
                 "budget, d0=0.08, F3 features): (V5.2 QT joint_best) vs "
                 "(R3-only checkpoint of the SAME run pair)",
        "folds_seeds": "dev folds 0-2 x seeds {17,29,43}; confirm folds "
                       "3-4 only when QT earned a pre-registered "
                       "dev-positive",
        "metrics": [
            "per-curve RMSE/zero over zero-step baseline (primary)",
            "SRE (frozen version), cos, sign accuracy, dead-zone rate",
            "strong/dead bucket diagnostics (R3 definitions, unchanged)",
        ],
        "aggregation": "independent curve/cluster unit (campaign "
                       "metrics_frozen.aggregate_unit); same curve "
                       "fwd/rev counts as ONE unit",
        "pass_condition": "RMSE/zero degradation <= 2% vs matched R3 "
                          "(campaign guards.rmse_degradation_max_pct); "
                          "sign drop <= 0.01; deadzone rise <= 0.01",
        "failure_action": "joint_fail -> reduce head lambda_q/lambda_stop "
                          "or keep the encoder frozen (Stage A only); "
                          "NEVER re-tune on confirm folds",
        "regression_guard": "Stop must NOT force MoveHead output to 0 near "
                            "TS (plan §5.6 #6): first-stop diagnostics "
                            "reported next to Move metrics, separately",
    },
    "inputs_required": [
        "R3 baseline run_records + checkpoints (G2 evidence)",
        "V5.2 run_records (E5) with checkpoint_update bindings",
        "frozen split_manifest_r3.json + this label report (class weights)",
    ],
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _quantiles(values: np.ndarray) -> dict:
    if values.size == 0:
        return {}
    qs = np.quantile(values, [0.1, 0.5, 0.9])
    return {
        "min": float(values.min()), "max": float(values.max()),
        "mean": float(values.mean()), "std": float(values.std()),
        "p10": float(qs[0]), "p50": float(qs[1]), "p90": float(qs[2]),
    }


def _load_dataset(tables_dir: Path) -> KeyAtomDatasetV3:
    return KeyAtomDatasetV3(
        tables_dir,
        label_statuses=("ok", "masked", "terminal", "conflict"))


def _integrity_checks(
    labels: V52LabelTable,
    dev_folds: Sequence[int],
) -> list[dict]:
    """Label-integrity checks (each returns pass/fail + evidence)."""
    checks: list[dict] = []

    def add(cid: str, ok: bool, evidence: object) -> None:
        checks.append({"id": cid, "pass": bool(ok), "evidence": evidence})

    # --- q range + anchors ---------------------------------------------------
    fwd_ok = [l for l in labels.by_sample.values()
              if l.q_mask and l.direction == "forward"]
    q_vals = np.asarray([l.q for l in fwd_ok], dtype=np.float64)
    add("q_in_range", bool(np.all(np.abs(q_vals) <= 1.0 + 1e-9)),
        {"max_abs_q": float(np.abs(q_vals).max()) if q_vals.size else None})

    anchors_bad = []
    for cid, c in labels.curves.items():
        if c.q_canonical is None:
            continue
        q = c.q_canonical
        ts_pos = c.node_order.index(c.ts_index)
        for name, got, want in (("A", q[0], -1.0), ("TS", q[ts_pos], 0.0),
                                ("B", q[-1], 1.0)):
            if abs(float(got) - want) > 1e-9:
                anchors_bad.append(
                    {"curve_id": cid, "anchor": name, "got": float(got)})
    add("q_anchors_exact", not anchors_bad,
        {"bad": anchors_bad[:10], "n_bad": len(anchors_bad)})

    # --- reverse q = -q (exact) ---------------------------------------------
    fq = {(l.curve_id, l.node_index): l.q for l in fwd_ok}
    rev = [l for l in labels.by_sample.values()
           if l.q_mask and l.direction == "reverse"]
    max_abs = 0.0
    n_pairs = 0
    for l in rev:
        key = (l.curve_id, l.node_index)
        if key in fq:
            n_pairs += 1
            max_abs = max(max_abs, abs(float(l.q) + float(fq[key])))
    add("reverse_q_negation_exact", max_abs <= 1e-12 and n_pairs > 0,
        {"pairs": n_pairs, "max_abs_q_fwd_plus_q_rev": max_abs})

    # --- monotonicity along path (diagnostic; fwd, per curve) ---------------
    nonmono = 0
    for cid, c in labels.curves.items():
        if c.q_canonical is None:
            continue
        q = np.asarray(c.q_canonical, dtype=np.float64)
        if bool(np.any(np.diff(q) < -1e-9)):
            nonmono += 1
    add("q_path_monotone_diagnostic", True,   # diagnostic only, never a gate
        {"curves_nonmonotone": nonmono, "note": "diagnostic, not gated"})

    # --- stop labels domain --------------------------------------------------
    stops = {l.stop for l in labels.by_sample.values()}
    add("stop_domain_1_0_mask", stops <= {1, 0, -1}, {"observed": sorted(stops)})
    return checks


def _stop_coverage_and_weights(
    labels: V52LabelTable,
    dev_folds: Sequence[int],
    confirm_folds: Sequence[int],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Per-fold stop coverage + frozen inner-train class weights."""
    by_fold_all: dict[int, Counter] = defaultdict(Counter)
    by_fold_ok: dict[int, Counter] = defaultdict(Counter)
    ok_pos_curves: dict[int, set] = defaultdict(set)
    for l in labels.by_sample.values():
        if l.fold is None:
            continue
        key = {1: "pos", 0: "neg", -1: "mask"}[l.stop]
        by_fold_all[int(l.fold)][key] += 1
        by_fold_all[int(l.fold)]["n"] += 1
        if l.label_status == "ok":
            by_fold_ok[int(l.fold)][key] += 1
            by_fold_ok[int(l.fold)]["n"] += 1
            if l.stop == 1:
                ok_pos_curves[int(l.fold)].add(l.curve_id)

    coverage: list[dict] = []
    for fold in sorted(set(by_fold_all) | set(by_fold_ok)):
        a, o = by_fold_all[fold], by_fold_ok[fold]
        coverage.append({
            "fold": fold,
            "all": {"n": a["n"], "pos": a["pos"], "neg": a["neg"],
                    "mask": a["mask"]},
            "ok": {"n": o["n"], "pos": o["pos"], "neg": o["neg"],
                   "mask": o["mask"],
                   "pos_rate": (o["pos"] / o["n"]) if o["n"] else None,
                   "mask_rate": (o["mask"] / o["n"]) if o["n"] else None},
            "ok_pos_curves": len(ok_pos_curves[fold]),
        })

    # --- frozen class weights (inner-train ONLY) ---------------------------
    dev = tuple(int(f) for f in dev_folds)
    weights: list[dict] = []
    insufficient: list[dict] = []

    def _inner_train_rows(fold_set: tuple[int, ...]):
        for l in labels.by_sample.values():
            if l.label_status != "ok" or l.fold is None:
                continue
            if int(l.fold) not in fold_set or not l.stop_mask:
                continue
            yield l

    def _estimate(fold_set: tuple[int, ...], source: str) -> dict:
        n_pos = n_neg = 0
        curves: set[str] = set()
        for l in _inner_train_rows(fold_set):
            if l.stop == 1:
                n_pos += 1
                curves.add(l.curve_id)
            else:
                n_neg += 1
        pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0
        return {"source": source, "folds": list(fold_set),
                "n_pos": n_pos, "n_neg": n_neg,
                "pos_weight": pos_weight, "n_pos_curves": len(curves)}

    for f in dev:
        inner = tuple(x for x in dev if x != f)
        rec = _estimate(inner, f"inner-train = dev folds minus fold {f}")
        rec["fold"] = f
        rec["frozen"] = True
        weights.append(rec)
        if (rec["n_pos"] < INSUFFICIENT_MIN_POS
                or rec["n_pos_curves"] < INSUFFICIENT_MIN_POS_CURVES):
            insufficient.append({
                "fold": f, "n_pos": rec["n_pos"],
                "n_pos_curves": rec["n_pos_curves"],
                "rule": {"min_pos": INSUFFICIENT_MIN_POS,
                         "min_pos_curves": INSUFFICIENT_MIN_POS_CURVES},
                "consequence": "Stop exploratory-only for this fold "
                               "(insufficient_stop_positives); windows NOT "
                               "widened"})
    conf = _estimate(dev, "inner-train = all dev folds (confirm runs)")
    conf["folds"] = list(dev)
    conf["confirm_for"] = list(confirm_folds)
    conf["frozen"] = True
    weights.append(conf)
    if (conf["n_pos"] < INSUFFICIENT_MIN_POS
            or conf["n_pos_curves"] < INSUFFICIENT_MIN_POS_CURVES):
        insufficient.append({
            "fold": "confirm", "n_pos": conf["n_pos"],
            "n_pos_curves": conf["n_pos_curves"],
            "rule": {"min_pos": INSUFFICIENT_MIN_POS,
                     "min_pos_curves": INSUFFICIENT_MIN_POS_CURVES},
            "consequence": "Stop exploratory-only on confirm folds "
                           "(insufficient_stop_positives)"})
    return coverage, weights, insufficient


def _q_stats_section(
    labels: V52LabelTable,
    dev_folds: Sequence[int],
) -> dict:
    per_fold = []
    for fold in sorted({int(l.fold) for l in labels.by_sample.values()
                        if l.fold is not None}):
        ok = [l for l in labels.by_sample.values()
              if l.fold == fold and l.label_status == "ok"]
        valid = np.asarray([l.q for l in ok if l.q_mask], dtype=np.float64)
        per_fold.append({
            "fold": fold,
            "n_ok": len(ok),
            "q_valid": int(valid.size),
            "q_masked": len(ok) - int(valid.size),
            "q_stats": _quantiles(valid),
        })
    total_valid = sum(e["q_valid"] for e in per_fold)
    return {
        "definition": Q_DEFINITION,
        "mass_table": "ATOMIC_MASS_TABLE_V52 (amu, Z=1..118, frozen E4)",
        "zero_arc_eps_angstrom": ZERO_ARC_EPS_ANGSTROM_V52,
        "zero_arc_curves": list(labels.zero_arc_curves),
        "n_zero_arc_curves": len(labels.zero_arc_curves),
        "masked_samples": len(labels.q_masked_sample_ids()),
        "coverage": {"q_valid": total_valid,
                     "q_masked": sum(e["q_masked"] for e in per_fold)},
        "per_fold": per_fold,
    }


def _head_wiring_section(dataset: KeyAtomDatasetV3) -> dict:
    """Param counts + shapes + bitwise backbone-unchanged (fixture + real)."""
    import torch

    from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3
    from pes2ts_core.pesn.models.key_atom_egnn_v3 import (
        MEASURED_PARAMS_V3,
        StarEGNNV3,
    )
    from pes2ts_core.pesn.models.key_atom_heads_v3 import (
        MEASURED_PARAMS_HEADS_V52,
        MEASURED_PARAMS_PROGRESS_HEAD_V52,
        MEASURED_PARAMS_TS_STOP_HEAD_V52,
        KeyAtomV52Model,
    )
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
        build_feature_batches,
    )

    torch.manual_seed(17)
    backbone = StarEGNNV3()
    model = KeyAtomV52Model(backbone)
    counts = model.head_parameter_counts()
    assert counts["backbone"] == MEASURED_PARAMS_V3
    assert counts["progress_head"] == MEASURED_PARAMS_PROGRESS_HEAD_V52
    assert counts["ts_stop_head"] == MEASURED_PARAMS_TS_STOP_HEAD_V52
    assert model.n_parameters == MEASURED_PARAMS_V3 + MEASURED_PARAMS_HEADS_V52

    # backbone weights untouched by wrapper construction
    state_before = {k: v.clone() for k, v in backbone.state_dict().items()}

    # real-data batch (first two ok samples of the first fold present)
    ok_idx = [i for i, s in enumerate(dataset.samples)
              if s.label_status == "ok"][:2]
    batch = build_feature_batches(dataset, ok_idx)
    fwd = forward_view_v3(batch)
    model.eval()
    with torch.no_grad():
        out = model(fwd)
        out_move = model.forward_move(fwd)
        out_bb = backbone(fwd)
    u_bitwise = torch.equal(out["u"], out_bb["u"]) \
        and torch.equal(out["u"], out_move["u"])
    z_bitwise = torch.equal(out["z"], out_bb["z"])
    state_after = backbone.state_dict()
    weights_unchanged = all(
        torch.equal(state_before[k], state_after[k]) for k in state_before)
    n_graphs = int(batch["cv_target_delta"].shape[0])
    ok_shapes = (
        tuple(out["u"].shape) == (int(batch["n_keys"].sum()), 3)
        and tuple(out["z"].shape) == (n_graphs, 64)
        and tuple(out["q"].shape) == (n_graphs,)
        and tuple(out["stop_logit"].shape) == (n_graphs,))
    return {
        "backbone_params": MEASURED_PARAMS_V3,
        "progress_head_params": counts["progress_head"],
        "ts_stop_head_params": counts["ts_stop_head"],
        "heads_params_total": (counts["progress_head"]
                               + counts["ts_stop_head"]),
        "joint_params": model.n_parameters,
        "shapes": {"u": "[ΣK,3]", "z": "[G,64]", "q": "[G]",
                   "stop_logit": "[G]"},
        "real_batch_n_graphs": n_graphs,
        "real_batch_shapes_ok": bool(ok_shapes),
        "u_bitwise_backbone_unchanged": bool(u_bitwise and z_bitwise),
        "backbone_weights_unchanged_after_attach": bool(weights_unchanged),
        "progress_head_arch": "tanh(Linear(64,32)->SiLU->Linear(32,1))",
        "ts_stop_head_arch": "Linear(64,1) raw logit (BCEWithLogitsLoss)",
    }


def build_e4_label_report(
    tables_dir: str | Path,
    *,
    dev_folds: Sequence[int] = DEV_FOLDS,
    confirm_folds: Sequence[int] = CONFIRM_FOLDS,
) -> dict:
    """Build the full E4/G3a label report dict from a six-table directory."""
    tables_dir = Path(tables_dir)
    dataset = _load_dataset(tables_dir)
    labels = dataset.v52_labels()

    table_shas = {
        name: _sha256(tables_dir / f"{name}.csv")
        for name in ("irc_curve", "irc_node", "irc_edge",
                     "reaction_intent", "curve_quality", "sample_view")}
    manifest_shas = {}
    for mf in ("data_manifest.json", "teacher_manifest.json",
               "split_manifest_r3.json"):
        p = tables_dir / mf
        if p.is_file():
            manifest_shas[mf] = _sha256(p)
    teacher_version = None
    tp = tables_dir / "teacher_manifest.json"
    if tp.is_file():
        teacher_version = json.loads(tp.read_text(encoding="utf-8")).get(
            "teacher_version")

    by_status = Counter(l.label_status for l in labels.by_sample.values())
    n_curves = len({l.curve_id for l in labels.by_sample.values()})

    integrity = _integrity_checks(labels, dev_folds)
    coverage, weights, insufficient = _stop_coverage_and_weights(
        labels, dev_folds, confirm_folds)
    q_section = _q_stats_section(labels, dev_folds)
    wiring = _head_wiring_section(dataset)

    integrity_pass = all(c["pass"] for c in integrity
                         if c["id"] != "q_path_monotone_diagnostic")
    wiring_pass = (wiring["u_bitwise_backbone_unchanged"]
                   and wiring["backbone_weights_unchanged_after_attach"]
                   and wiring["real_batch_shapes_ok"])
    if not integrity_pass or not wiring_pass:
        proposed_state = "fail"
    elif insufficient:
        proposed_state = "partial"     # Stop exploratory-only (plan §2.1)
    else:
        proposed_state = "pass"

    return {
        "schema": REPORT_SCHEMA,
        "label_schema": STOP_LABEL_SCHEMA_V52,
        "campaign_id": "PES2TS_30H_R3_V52_V53",
        "generated_at": datetime.now(timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "tables_dir": str(tables_dir),
        "provenance": {
            "table_sha256": table_shas,
            "manifest_sha256": manifest_shas,
            "teacher_version": teacher_version,
            "truth_reads": 0,
            "note": "labels derive ONLY from six-table teacher columns",
        },
        "dataset": {
            "n_samples": len(labels.by_sample),
            "n_curves": n_curves,
            "by_status": dict(sorted(by_status.items())),
        },
        "q_labels": q_section,
        "stop_labels": {
            "definition": STOP_DEFINITION,
            "windows_angstrom": {"positive_le": 0.04, "negative_ge": 0.12,
                                 "middle": "mask"},
            "coverage_per_fold": coverage,
            "insufficient_stop_positives": {
                "flagged": bool(insufficient),
                "folds": insufficient,
                "rule_constants": {
                    "min_pos": INSUFFICIENT_MIN_POS,
                    "min_pos_curves": INSUFFICIENT_MIN_POS_CURVES,
                    "frozen": False,
                    "note": "engineering precheck (plan §5.3 has no frozen "
                            "threshold); reported, never used to widen "
                            "windows or fabricate positives",
                },
            },
            "class_weights_frozen": weights,
        },
        "head_wiring": wiring,
        "integrity_checks": integrity,
        "move_non_degradation": MOVE_NON_DEGRADATION_DESIGN,
        "gate": {
            "gate": "G3a-Label",
            "proposed_state": proposed_state,
            "integrity_pass": integrity_pass,
            "wiring_pass": bool(wiring_pass),
            "stop_insufficient_positives": bool(insufficient),
            "note": "proposed_state is evidence for the supervisor-bound "
                    "G3a decision (gate_guard); H12: without an R3 "
                    "baseline only contract/label claims are admissible",
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="e4_label_gate")
    parser.add_argument("--tables-dir", default=None,
                        help="E1 six-table directory (real data)")
    parser.add_argument("--report", required=True,
                        help="output JSON path (e.g. outputs/pesn/"
                             "30h_campaign/e4_labels/e4_label_report.json)")
    parser.add_argument("--fixture", action="store_true",
                        help="run on the deterministic v3_fixtures tables "
                             "(mechanics check; NOT admissible evidence)")
    args = parser.parse_args(argv)

    if args.fixture:
        import tempfile

        from scripts.agent_ops.feature_gate.v3_fixtures import (
            write_fixture_tables,
        )
        with tempfile.TemporaryDirectory() as td:
            write_fixture_tables(td, n_frames=9, n_curves=2, fold=0)
            report = build_e4_label_report(td)
        report["fixture_mode"] = True
        report["admissibility"] = "fixture_only_not_admissible"
    else:
        if not args.tables_dir:
            parser.error("--tables-dir is required unless --fixture")
        report = build_e4_label_report(args.tables_dir)
        report["fixture_mode"] = False

    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print(json.dumps({
        "report": str(out),
        "proposed_state": report["gate"]["proposed_state"],
        "n_samples": report["dataset"]["n_samples"],
        "stop_insufficient_positives":
            report["stop_labels"]["insufficient_stop_positives"]["flagged"],
        "zero_arc_curves": report["q_labels"]["n_zero_arc_curves"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
