#!/usr/bin/env python3
"""V5.4.1 H0 legality audit — fixture/restore level (plan §11.1 H0, §7.4).

Executes the preregistered H0 gate as REAL checks on a synthetic PD-1
fixture (never the 50K pool, never dev/confirm, no training, no QC) and
writes four machine artifacts under ``<root>/audits``:

``h0_legality.json``           per-item pass/fail + evidence SHAs + the
                               single ``passed`` verdict (fail-closed)
``geometry_invariance.json``   E(3) / permutation / scalar / batch
                               consistency per candidate at a NON-zero gate
``input_leak.json``            teacher & metadata intervention, forward
                               whitelist, non-key-atom action, finite
                               outputs+grads, E2 degree stats, E6 refusal
``resume_equivalence.json``    E0 20+20 vs 40 bitwise, E1–E5 2+2 vs 4,
                               identity / content-digest typed refusals

H0 rule (frozen config ``gates.H0``): schema / no-leak / confirm isolation
/ restore / E(3) / permutation / non-key-atom action must ALL pass.  This
module is deliberately FAIL-CLOSED: a section that raises, or an evidence
file that is missing, is recorded as a FAILED item — ``passed`` is ``True``
only when every item is ``True``.  Nothing here grants a gate; it only
produces the evidence the runner reads (``v541_run.t1_gate_report``).

Boundaries (hard):
* no model / provider / runner / config / doc is modified — everything is
  imported and CALLED;
* no real 50K training, no QC call, no dev/confirm read: the dataset is a
  synthetic six-table fixture written into ``audits/h0_work/tables``;
* the E2 degree-statistics fit goes through the reviewed
  ``v541_run.ensure_degree_stats`` helper, but its cache is re-anchored on
  the audit work dir so a fixture fit can never pollute the campaign cache;
* real GPU 32-sample forward/backward belongs to the 200-update smoke, not
  here (H0 is fixture/restore level; CPU only).

CLI::

    python scripts/campaigns/v541_legality_audit.py \\
        [--root outputs/pesn/v541_embed_50k] [--config PATH] [--force]

Exit codes: 0 = H0 passed; 1 = H0 failed (fail-closed, not granted);
2 = refused (artifacts exist without ``--force``).
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np                                          # noqa: E402
import torch                                                # noqa: E402

from pes2ts_core.pesn.contracts.key_atom_v3 import (        # noqa: E402
    SUPERVISION_KEYS_V3,
    KeyAtomV3ContractError,
    forward_view_v3,
)
from pes2ts_core.pesn.features.key_atom_graph_v3 import (   # noqa: E402
    build_sample_features_v3,
    collate_v3,
    to_torch_batch,
)
from pes2ts_core.pesn.models.v541 import (                  # noqa: E402
    V541Model,
)
from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (  # noqa: E402
    KeyAtomDatasetV3,
)
from pes2ts_core.pesn.offline.training import key_atom_v53mp_train_v3 as tr  # noqa: E402
from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import (   # noqa: E402
    set_seed,
    sha256_file,
)
from pes2ts_core.pesn.offline.training.v541_degree_stats import (  # noqa: E402
    ALLOWED_SPLIT as DEGREE_STATS_SPLIT,
    compute_degree_stats,
)
from pes2ts_core.pesn.offline.training.v541_identity import (  # noqa: E402
    git_state,
    positions_sha256,
    read_manifest,
)
from pes2ts_core.pesn.offline.training.v541_monitor import (  # noqa: E402
    host_peak_rss_bytes,
)
from pes2ts_core.pesn.offline.training.v541_train import (   # noqa: E402
    V541TrainerConfig,
    V541TrainerError,
    degree_stats_payload_digest,
    load_degree_stats_v541,
    load_model_v541,
    set_training_stage_v541,
    train_run_v541,
)
from scripts.agent_ops.feature_gate.equivariance_harness import (  # noqa: E402
    _random_rotation,
    check_permutation_equivariance,
    check_rotation_equivariance,
    check_translation_invariance,
)
from scripts.agent_ops.feature_gate.v3_fixtures import (     # noqa: E402
    ATOMIC_NUMBERS_SN2,
    BASE_XYZ_SN2,
    make_sn2_intent,
    make_sn2_sample,
    make_sn2_teacher,
    make_v3_batch,
    make_v3_gate_batch,
)
from scripts.agent_ops.v541_config_schema import (           # noqa: E402
    FORMAL_THRESHOLDS,
    load_config,
    validate_config,
)
from scripts.campaigns.v541_run import (                     # noqa: E402
    RunInputs,
    ensure_degree_stats,
)

__all__ = [
    "AUDIT_ALPHA_V541",
    "CANDIDATES_V541",
    "H0_LEGITIMACY_SCHEMA",
    "build_fixture_env",
    "main",
    "run_h0_legality_audit",
]

# ---------------------------------------------------------------------------
# frozen audit constants
# ---------------------------------------------------------------------------

H0_LEGITIMACY_SCHEMA = "pes2ts_v541_h0_legality_audit_v1"
GEO_SCHEMA = "pes2ts_v541_geometry_invariance_audit_v1"
LEAK_SCHEMA = "pes2ts_v541_input_leak_audit_v1"
RESUME_SCHEMA = "pes2ts_v541_resume_equivalence_audit_v1"

#: Main-matrix candidates H0 must clear (E6 is a refusal check, §6.1).
CANDIDATES_V541: Tuple[str, ...] = ("E0", "E1", "E2", "E3", "E4", "E5")

#: Non-zero gate used for every geometry/leak check (plan §7.5: alpha=0
#: makes a branch look free; H0 must see the gate OPEN).
AUDIT_ALPHA_V541 = 0.25

AUDIT_SEED_V541 = 17
#: Independent branch-init seed (§5.4: never the run seed).
AUDIT_INIT_SEED_V541 = 20261028

#: Normative E(3)/permutation tolerance (PD-2/PD-6, gate-admissible).
EQUIV_PRESET_V541 = "30h_v1.1_normative"
EQUIV_ABS_V541 = 1.0e-6
EQUIV_REL_V541 = 1.0e-5

#: Fixture geometry: pool (fold 1) + dev (fold 0) + confirm (fold 3) so the
#: split / firewall code paths are exercised exactly as on real data.
FIXTURE_FOLDS_V541 = {
    "cv0": 1, "cv1": 1, "cv2": 1,
    "cv_dev0": 0, "cv_dev1": 0,
    "cv_confirm": 3,
}
N_NODES_V541 = 7
SCALAR_OUTPUT_KEYS_V541 = ("z", "q", "endpoint_logit", "order_score")

DEFAULT_ROOT = "outputs/pesn/v541_embed_50k"


class H0AuditError(RuntimeError):
    """Typed refusal of the H0 audit tooling (never a silent skip)."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = str(code)


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise H0AuditError(code, detail)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha(obj: Any) -> str:
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _check(name: str, ok: bool, **evidence: Any) -> Dict[str, Any]:
    return {"check": name, "passed": bool(ok), **evidence}


# ---------------------------------------------------------------------------
# synthetic PD-1 fixture (pool + dev + confirm folds; no real data)
# ---------------------------------------------------------------------------


def _write_csv(path: Path, header, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(list(header))
        for row in rows:
            writer.writerow(list(row))


def _curve_frames(curve_idx: int, n_nodes: int) -> list:
    """Deterministic monotone SN2 path (Br approaches C, O approaches C)."""
    rng = np.random.default_rng(1234 + int(curve_idx))
    frames = []
    for k in range(n_nodes):
        t = k / max(n_nodes - 1, 1)
        xyz = np.array(BASE_XYZ_SN2, dtype=np.float64)
        xyz[1, 0] -= 1.1 * t
        xyz[2, 0] += 1.1 * t
        xyz += rng.normal(scale=0.01, size=xyz.shape)
        frames.append(xyz)
    return frames


def write_fixture_tables(tables_dir: Path,
                         folds: Mapping[str, int]) -> None:
    """Write the PD-1 six-table layout for ``folds`` (multi-curve)."""
    out = Path(tables_dir)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json = json.dumps(list(ATOMIC_NUMBERS_SN2))
    curve_rows, node_rows, intent_rows, edge_rows, quality_rows, sample_rows = \
        [], [], [], [], [], []
    for curve_idx, (cid, fold) in enumerate(sorted(folds.items())):
        fwd = make_sn2_intent("forward", curve_id=cid)
        rev = fwd.reversed_intent()
        frames = _curve_frames(curve_idx, N_NODES_V541)
        ts_index = N_NODES_V541 // 2
        curve_rows.append((cid, "canon::{}".format(cid), N_NODES_V541,
                           2 * 4, ts_index, 0, N_NODES_V541 - 1))
        for frame, xyz in enumerate(frames):
            node_rows.append((
                cid, frame, json.dumps(xyz.tolist()), 0.0, 0.0,
                -1.0 + 2.0 * frame / max(N_NODES_V541 - 1, 1),
                frame == ts_index, frame in (0, N_NODES_V541 - 1),
                atomic_json))
        for it in (fwd, rev):
            intent_rows.append((
                it.intent_id, cid, it.direction, it.reverse_of_intent_id,
                it.intent_pair_id, "false",
                json.dumps([list(b) for b in it.g_src_bonds]),
                json.dumps([list(b) for b in it.g_tgt_bonds]),
                "[]", json.dumps({"target_deltas": list(it.cv_deltas)}),
                "", "", "v3"))
            for edge_index, (src, dst) in enumerate(
                    ((0, 1), (1, 2), (2, 3), (3, 4))):
                flat = list(np.asarray(
                    make_sn2_teacher(it.direction), dtype=np.float64).ravel())
                packed = hashlib.sha256(b"fixture-atom-map").hexdigest()
                edge_rows.append((
                    cid, it.intent_id, edge_index, src, dst,
                    json.dumps(flat), "[6,3]", f"[0,{len(flat)})",
                    packed,
                    hashlib.sha256(json.dumps(flat).encode()).hexdigest(),
                    "angstrom", "true"))
        quality_rows.append((cid, "ok"))
        n_samples = N_NODES_V541 * 2
        cw = 1.0 / n_samples
        for frame in range(N_NODES_V541):
            for it in (fwd, rev):
                target = json.dumps(
                    [list(r) for r in make_sn2_teacher(it.direction)])
                sample_rows.append((
                    f"{cid}:{frame}:{it.intent_id}:0.08", cid, frame,
                    it.intent_id, 0.08, 0.0, 0.0, 0.0, 1.0, cw, 1.0, "ok",
                    hashlib.sha256(b"feature").hexdigest(),
                    hashlib.sha256(b"label").hexdigest(),
                    "false", "local_frame", len(it.key_rows()) - 1,
                    target, 0.0, 0, "false", int(fold)))

    _write_csv(out / "irc_curve.csv",
               ("curve_id", "canonical_irc_id", "n_nodes", "n_edges",
                "ts_index", "endpoint_a_index", "endpoint_b_index"),
               curve_rows)
    _write_csv(out / "irc_node.csv",
               ("curve_id", "node_index", "x", "y", "z", "progress",
                "is_ts", "is_endpoint", "atomic_numbers"), node_rows)
    _write_csv(out / "reaction_intent.csv",
               ("intent_id", "curve_id", "direction", "reverse_of_intent_id",
                "intent_pair_id", "intent_underdetermined", "g_src", "g_tgt",
                "delta_b_intent", "cv_spec", "intent_role", "rules",
                "intent_version"), intent_rows)
    _write_csv(out / "irc_edge.csv",
               ("curve_id", "intent_id", "edge_index", "src_node_index",
                "dst_node_index", "delta_xyz_flat", "delta_xyz_shape",
                "delta_xyz_offsets", "atom_map_sha256", "delta_xyz_sha256",
                "units", "same_ref_frame"), edge_rows)
    _write_csv(out / "curve_quality.csv",
               ("curve_id", "quality_status"), quality_rows)
    _write_csv(out / "sample_view.csv",
               ("sample_id", "curve_id", "node_index", "intent_id", "d0",
                "x", "y", "z", "label_weight", "curve_weight",
                "sample_weight", "label_status", "feature_sha", "label_sha",
                "sensitivity_only", "gauge_ref", "target_node_index",
                "target_action_u", "target_progress_q", "stop_label", "mask",
                "fold"), sample_rows)
    names = ("irc_curve.csv", "irc_node.csv", "irc_edge.csv",
             "reaction_intent.csv", "curve_quality.csv", "sample_view.csv")
    (out / "data_manifest.json").write_text(json.dumps(
        {"files": [{"path": n, "sha256": sha256_file(out / n)}
                   for n in names]},
        indent=2, sort_keys=True) + "\n", encoding="utf-8")


@dataclass
class FixtureEnv:
    """Synthetic dataset + labels + frozen dev/confirm/train position sets."""

    ds: Any
    labels: Any
    endpoint: Any
    dev_pos: List[int]
    pool_pos: List[int]
    confirm_pos: List[int]
    dev_curve_ids: List[str]
    confirm_curve_ids: List[str]
    train_pos: List[int]
    inner_val_pos: List[int]
    tables_dir: Path


@dataclass
class _EpRow:
    endpoint_label: int
    d_end_angstrom: float
    bond_edit_complete: bool
    geometry_legal: bool
    ts_row: bool
    start_row: bool
    terminal_row: bool
    degenerate_curve: bool


class _EpTable:
    def __init__(self, by_sample: Mapping[str, Any]) -> None:
        self.by_sample = by_sample


def _build_endpoint_table(dataset: Any) -> _EpTable:
    rows = {}
    for rec in dataset.samples:
        node = int(rec.node_index)
        fwd = rec.intent.direction == "forward"
        end_node = N_NODES_V541 - 1 if fwd else 0
        start_node = 0 if fwd else N_NODES_V541 - 1
        ts_node = dataset.curve_ts_index(rec.curve_id)
        label = 0
        if node == ts_node:
            label = 0
        elif node == end_node:
            label = 1
        elif node == start_node:
            label = 0
        rows[rec.sample_id] = _EpRow(
            endpoint_label=label,
            d_end_angstrom=0.15 * abs(node - end_node),
            bond_edit_complete=(label == 1),
            geometry_legal=True,
            ts_row=(node == ts_node),
            start_row=(node == start_node),
            terminal_row=False,
            degenerate_curve=False,
        )
    return _EpTable(rows)


def build_fixture_env(tables_dir: Path) -> FixtureEnv:
    """Load the synthetic fixture and derive the frozen split position sets."""
    tables_dir = Path(tables_dir)
    write_fixture_tables(tables_dir, FIXTURE_FOLDS_V541)
    dataset = KeyAtomDatasetV3(tables_dir, label_statuses=("ok",))
    labels = dataset.v52_labels()
    endpoint = _build_endpoint_table(dataset)
    samples = dataset.samples

    def _pos(folds: Sequence[int]) -> List[int]:
        return [i for i, r in enumerate(samples)
                if int(getattr(r, "fold", -1)) in folds]

    pool_pos = _pos((1,))
    dev_pos = _pos((0,))
    confirm_pos = _pos((3,))
    pool_curves = sorted({str(samples[i].curve_id) for i in pool_pos})
    train_pos, inner_val_pos = tr.v53mp_pool_split(
        samples, pool_curve_ids=pool_curves, seed=AUDIT_SEED_V541,
        inner_val_fraction=0.2)
    return FixtureEnv(
        ds=dataset, labels=labels, endpoint=endpoint, dev_pos=dev_pos,
        pool_pos=pool_pos, confirm_pos=confirm_pos,
        dev_curve_ids=sorted({str(samples[i].curve_id) for i in dev_pos}),
        confirm_curve_ids=sorted(
            {str(samples[i].curve_id) for i in confirm_pos}),
        train_pos=train_pos, inner_val_pos=inner_val_pos,
        tables_dir=tables_dir)


# ---------------------------------------------------------------------------
# model / config helpers
# ---------------------------------------------------------------------------


def audit_config(embedding_id: str, *,
                 degree_stats: Optional[Mapping[str, Any]] = None,
                 **over: Any) -> V541TrainerConfig:
    """A short-schedule audit config (fixture level; NEVER a formal run)."""
    base: Dict[str, Any] = dict(
        seed=AUDIT_SEED_V541, backbone_init="scratch", device="cpu",
        freeze_encoder_updates=1, joint_updates=3, smoke_updates=4,
        val_every=2, grad_monitor_every=2)
    if embedding_id != "E0":
        base["embedding_id"] = embedding_id
        base["init_seed"] = AUDIT_INIT_SEED_V541
    if degree_stats:
        base["degree_stats_path"] = str(degree_stats["path"])
        base["degree_stats_sha256"] = str(degree_stats["sha256"])
    base.update(over)
    return V541TrainerConfig(**base)


def build_audit_model(embedding_id: str,
                      degree_stats: Optional[Mapping[str, Any]] = None,
                      **over: Any) -> V541Model:
    """Construction through the RUNNER's own path (``load_model_v541``)."""
    set_seed(AUDIT_SEED_V541)
    cfg = audit_config(embedding_id, degree_stats=degree_stats, **over)
    return load_model_v541(cfg, device="cpu")


def open_gate(model: V541Model) -> Dict[str, Any]:
    """Force a NON-zero gate and report what was set (E0 has none)."""
    alpha = getattr(model, "alpha", None)
    if alpha is None:
        return {"alpha": None, "reason": "e0_no_branch",
                "requested": AUDIT_ALPHA_V541}
    with torch.no_grad():
        alpha.fill_(float(AUDIT_ALPHA_V541))
    return {"alpha": float(alpha.detach()), "reason": None,
            "requested": AUDIT_ALPHA_V541}


def _call(model: Any, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
    was_training = getattr(model, "training", None)
    if hasattr(model, "eval"):
        model.eval()
    try:
        with torch.no_grad():
            out = model(batch)
    finally:
        if was_training and hasattr(model, "train"):
            model.train()
    return {k: v for k, v in out.items() if isinstance(v, torch.Tensor)}


# ---------------------------------------------------------------------------
# degree stats (E2): ONE legal inner-train fit through the runner helper
# ---------------------------------------------------------------------------


def _audit_cfg_cache(cfg: Mapping[str, Any], cache_dir: Path) -> Dict[str, Any]:
    """Re-anchor the degree-stats cache on THIS audit's work dir.

    The frozen config declares a repo-relative cache root; the runner
    re-anchors it on ``--root``.  H0 must never write a fixture-derived fit
    into the real campaign cache, so the audit pins an ABSOLUTE path under
    ``audits/h0_work`` — the fit itself still goes through the reviewed
    ``v541_run.ensure_degree_stats`` unmodified.
    """
    out = copy.deepcopy(dict(cfg))
    recipe = ((out.get("embedding_contract") or {})
              .get("feature_recipe") or {})
    recipe["degree_stats"] = dict(recipe.get("degree_stats") or {})
    recipe["degree_stats"]["cache"] = str(Path(cache_dir).resolve())
    return out


def _fit_degree_stats(cfg: Mapping[str, Any], env: FixtureEnv, *,
                      cache_dir: Path, root: Path,
                      git: Optional[str]) -> Dict[str, Any]:
    """Inner-train-only E2 fit through the reviewed runner helper."""
    inputs = RunInputs(
        dataset=env.ds, labels=env.labels, endpoint=env.endpoint,
        split_doc={"curve_folds": FIXTURE_FOLDS_V541},
        pool_curve_ids=sorted({str(env.ds.samples[i].curve_id)
                               for i in env.pool_pos}),
        dev_curve_ids=list(env.dev_curve_ids),
        confirm_curve_ids=list(env.confirm_curve_ids),
        provenance={})
    return ensure_degree_stats(
        _audit_cfg_cache(cfg, cache_dir), embedding_id="E2", inputs=inputs,
        train_positions=list(env.train_pos), root=root,
        code_sha=git or "unknown", dev_positions=list(env.dev_pos),
        confirm_curve_ids=list(env.confirm_curve_ids),
        seed=AUDIT_SEED_V541)


# ---------------------------------------------------------------------------
# section 1 — schema / thresholds / data audit / confirm isolation
# ---------------------------------------------------------------------------


def audit_schema_and_gates(*, root: Path,
                           config_path: Optional[Path]) -> Dict[str, Any]:
    """§11.1 H0 first clause: schema, frozen thresholds, confirm firewall."""
    audits = root / "audits"
    items: List[Dict[str, Any]] = []

    cfg = load_config(config_path)
    report = validate_config(cfg, formal=True, check_gv=True)
    items.append(_check(
        "config_schema_formal",
        bool(report.get("ok")) and report.get("mode") == "formal",
        schema=report.get("schema"), campaign_id=report.get("campaign_id"),
        objective=report.get("objective"),
        t1_config_ready=report.get("t1_config_ready"),
        t1_blocked=report.get("t1_blocked")))

    signoff = (cfg.get("thresholds") or {}).get("signoff") or {}
    missing = [k for k in FORMAL_THRESHOLDS if signoff.get(k) is None]
    items.append(_check(
        "formal_thresholds_signed_frozen",
        str(signoff.get("status")) == "frozen" and not missing,
        status=signoff.get("status"), signed_by=signoff.get("signed_by"),
        signed_on=signoff.get("signed_on"),
        thresholds_present=sorted(k for k in FORMAL_THRESHOLDS
                                  if signoff.get(k) is not None),
        missing=missing))

    auth = (cfg.get("governance") or {}).get("authorization") or {}
    items.append(_check(
        "config_authorization_firewall",
        auth.get("confirm_authorized") is False
        and auth.get("sealed_authorized") is False
        and int(auth.get("new_qc_calls", -1)) == 0,
        confirm_authorized=auth.get("confirm_authorized"),
        sealed_authorized=auth.get("sealed_authorized"),
        new_qc_calls=auth.get("new_qc_calls")))

    items.append(_check(
        "single_objective_no_arm_matrix",
        cfg.get("objective") == "unified_move_process"
        and not any(k in cfg for k in ("arm", "arms", "arm_matrix")),
        objective=cfg.get("objective"),
        arm_matrix_keys=[k for k in ("arm", "arms", "arm_matrix") if k in cfg]))

    # --- data audit produced by the data lane (read-only here) -----------
    data_path = audits / "data_audit.json"
    if data_path.is_file():
        doc = json.loads(data_path.read_text(encoding="utf-8"))
        items.append(_check(
            "data_audit_ok",
            doc.get("ok") is True and not doc.get("failed"),
            path=str(data_path), sha256=sha256_file(data_path),
            audit_ok=doc.get("ok"), failed=doc.get("failed"),
            counts=doc.get("counts")))
        items.append(_check(
            "data_audit_confirm_isolated",
            doc.get("confirm_read") is False
            and int(doc.get("new_qc_calls", -1)) == 0
            and doc.get("dev_used_for") == "curve_exclusion_only",
            confirm_read=doc.get("confirm_read"),
            new_qc_calls=doc.get("new_qc_calls"),
            dev_used_for=doc.get("dev_used_for")))
    else:
        items.append(_check("data_audit_ok", False, path=str(data_path),
                            reason="evidence file missing (fail-closed)"))
        items.append(_check("data_audit_confirm_isolated", False,
                            path=str(data_path),
                            reason="evidence file missing (fail-closed)"))

    return {
        "schema": "pes2ts_v541_h0_schema_gates_v1",
        "items": items,
        "passed": all(i["passed"] for i in items),
        "config_path": str(config_path) if config_path else None,
        "generated_at": _utc_now(),
    }


# ---------------------------------------------------------------------------
# section 2 — geometry: E(3), permutation, scalar, batch consistency
# ---------------------------------------------------------------------------


def _reflection_matrix() -> torch.Tensor:
    return torch.diag(torch.tensor([1.0, 1.0, -1.0]))


def _scalar_report(model: Any, batch: Mapping[str, Any], *,
                   rotation: Optional[torch.Tensor] = None,
                   shift: Optional[torch.Tensor] = None) -> Dict[str, Any]:
    """Scalar outputs (z/q/endpoint/order) under an E(3) transform."""
    moved = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
             for k, v in batch.items()}
    if shift is not None:
        moved["pos"] = moved["pos"] + shift.to(moved["pos"].dtype)
    if rotation is not None:
        moved["pos"] = moved["pos"] @ rotation.to(moved["pos"].dtype).T
    a = _call(model, batch)
    b = _call(model, moved)
    rows: Dict[str, Any] = {}
    passed = True
    for key in SCALAR_OUTPUT_KEYS_V541:
        da, db = a[key], b[key]
        delta = (da - db).abs()
        scale = max(float(da.abs().max()), float(db.abs().max()))
        threshold = EQUIV_ABS_V541 + EQUIV_REL_V541 * scale
        ok = bool(delta.numel() and float(delta.max()) <= threshold)
        passed = passed and ok
        rows[key] = {
            "bitwise": bool(torch.equal(da, db)),
            "max_abs_delta": float(delta.max()) if delta.numel() else 0.0,
            "scale": scale, "threshold": threshold, "passed": ok,
        }
    return {"passed": passed, "scalars": rows}


def _split_per_graph(out: Mapping[str, torch.Tensor],
                     full: Mapping[str, Any]
                     ) -> List[Dict[str, torch.Tensor]]:
    """Per-graph views of ``u``/scalars using the FULL batch's ``n_keys``."""
    n_keys = [int(v) for v in torch.as_tensor(full["n_keys"]).tolist()]
    bounds = np.cumsum([0] + n_keys)
    per = []
    for i, _k in enumerate(n_keys):
        per.append({
            "u": out["u"][bounds[i]:bounds[i + 1]],
            "z": out["z"][i:i + 1],
            "q": out["q"][i:i + 1],
            "endpoint_logit": out["endpoint_logit"][i:i + 1],
            "order_score": out["order_score"][i:i + 1],
        })
    return per


def _batch_consistency_report(model: Any) -> Dict[str, Any]:
    """Independent graph vs the same graph inside a MIXED batch (§11.1).

    ``n_keys`` lives only on the FULL collated batch, so the model is fed
    ``forward_view_v3(full)`` while the split uses the full batch.
    """
    single_full = make_v3_batch(n_graphs=1, bidirectional=False)
    mixed_full = make_v3_batch(n_graphs=3)
    single = forward_view_v3(single_full)
    mixed = forward_view_v3(mixed_full)
    per_single = _split_per_graph(_call(model, single), single_full)
    per_mixed = _split_per_graph(_call(model, mixed), mixed_full)
    rows: Dict[str, Any] = {}
    passed = True
    for key in ("u", "z", "q", "endpoint_logit", "order_score"):
        deltas = []
        bitwise = True
        for g in range(len(per_single)):
            a = per_single[g][key]
            b = per_mixed[g][key]
            bitwise = bitwise and bool(torch.equal(a, b))
            deltas.append(float((a - b).abs().max()))
        scale = max(float(per_single[0][key].abs().max()), 1e-12)
        threshold = EQUIV_ABS_V541 + EQUIV_REL_V541 * scale
        ok = bool(deltas) and max(deltas) <= threshold
        passed = passed and ok
        rows[key] = {"max_abs_delta": max(deltas), "bitwise": bitwise,
                     "threshold": threshold, "passed": ok,
                     "n_graphs_compared": len(per_single)}
    return {"passed": passed, "rows": rows,
            "independent_graphs": len(per_single),
            "mixed_graphs": len(per_mixed)}


def _candidate_geometry(model: V541Model, gate: Mapping[str, Any]
                        ) -> Dict[str, Any]:
    """All geometry checks for ONE candidate on the canonical fixture batch."""
    batch = make_v3_gate_batch(n_graphs=2)
    shift = torch.tensor([1.7, -0.3, 2.2])
    rotation = _random_rotation(20261028)
    reflection = _reflection_matrix()

    translation = check_translation_invariance(
        model, batch, preset=EQUIV_PRESET_V541, shift=shift)
    rotation_rep = check_rotation_equivariance(
        model, batch, preset=EQUIV_PRESET_V541, rotation=rotation)
    permutation = check_permutation_equivariance(
        model, batch, preset=EQUIV_PRESET_V541, seed=AUDIT_SEED_V541)

    # reflection (improper, det = -1) — the harness only does proper R
    rot = {k: (v.clone() if isinstance(v, torch.Tensor) else v)
           for k, v in batch.items()}
    rot["pos"] = batch["pos"] @ reflection.T
    out_a = _call(model, batch)
    out_b = _call(model, rot)
    per_atom = ((out_a["u"] @ reflection.T) - out_b["u"]).norm(dim=-1)
    refl_scale = max(float(out_a["u"].abs().max()), 1e-12)
    refl_threshold = EQUIV_ABS_V541 + EQUIV_REL_V541 * refl_scale
    reflection_rep = {
        "check": "reflection_equivariance",
        "passed": bool(per_atom.numel()
                       and float(per_atom.max()) <= refl_threshold),
        "max": float(per_atom.max()) if per_atom.numel() else 0.0,
        "scale": refl_scale, "threshold": refl_threshold,
        "det": float(torch.det(reflection)),
        "per_atom_errors": per_atom.tolist(),
    }

    scalars = {
        "translation": _scalar_report(model, batch, shift=shift),
        "rotation": _scalar_report(model, batch, rotation=rotation),
        "reflection": _scalar_report(model, batch, rotation=reflection),
    }
    consistency = _batch_consistency_report(model)

    n_atoms = int(batch["pos"].shape[0])
    n_keys = int(batch["key_index"].numel())
    shape_ok = tuple(out_a["u"].shape) == (n_keys, 3) and n_keys < n_atoms

    passed = all((
        translation["passed"], rotation_rep["passed"],
        permutation["passed"], reflection_rep["passed"],
        all(s["passed"] for s in scalars.values()),
        consistency["passed"], shape_ok,
    ))
    return {
        "gate": dict(gate),
        "passed": passed,
        "preset": EQUIV_PRESET_V541,
        "translation_invariance_u": translation,
        "rotation_equivariance_u": rotation_rep,
        "reflection_equivariance_u": reflection_rep,
        "permutation_equivariance_u": permutation,
        "scalar_invariance": scalars,
        "batch_consistency": consistency,
        "non_key_atoms_no_direct_action": {
            "passed": shape_ok, "n_atoms": n_atoms, "n_keys": n_keys,
            "u_shape": list(out_a["u"].shape),
        },
    }


def audit_geometry(*, degree_stats_for: Mapping[str, Any]) -> Dict[str, Any]:
    """E(3) / permutation / scalars / batch consistency per candidate."""
    per_candidate: Dict[str, Any] = {}
    for eid in CANDIDATES_V541:
        model = build_audit_model(eid, degree_stats_for.get(eid))
        gate = open_gate(model)
        per_candidate[eid] = _candidate_geometry(model, gate)
    return {
        "schema": GEO_SCHEMA,
        "generated_at": _utc_now(),
        "alpha_policy": ("E0 has no gate; every candidate is audited with "
                         "alpha={}".format(AUDIT_ALPHA_V541)),
        "preset": EQUIV_PRESET_V541,
        "per_candidate": per_candidate,
        "passed": all(c["passed"] for c in per_candidate.values()),
    }


# ---------------------------------------------------------------------------
# section 3 — input leak / whitelist / non-key action / finite / E2 / E6
# ---------------------------------------------------------------------------


def _intervention_block(model: V541Model) -> Dict[str, Any]:
    """Teacher & metadata intervention must leave EVERY output identical."""
    samples = [make_sn2_sample(d, node_index=i, graph_index=0)
               for i, d in enumerate(("forward", "reverse"))]
    clean = [build_sample_features_v3(s, level="F3") for s in samples]
    perturbed = []
    for built in clean:
        row = dict(built)
        if row.get("y_teacher") is not None:
            row["y_teacher"] = np.asarray(
                row["y_teacher"], dtype=np.float32) + 0.875
        row["frame_weight"] = np.asarray(
            row.get("frame_weight", [1.0]), dtype=np.float32) * 0.125
        perturbed.append(row)

    fwd_clean = forward_view_v3(to_torch_batch(collate_v3(clean)))
    fwd_pert = forward_view_v3(to_torch_batch(collate_v3(perturbed)))
    view_identical = all(
        torch.equal(fwd_clean[k], fwd_pert[k])
        for k in fwd_clean if isinstance(fwd_clean[k], torch.Tensor))
    out_a = _call(model, fwd_clean)
    out_b = _call(model, fwd_pert)
    outputs_identical = all(torch.equal(out_a[k], out_b[k]) for k in out_a)

    # supervision keys must be REFUSED when they reach model.forward
    raw = to_torch_batch(collate_v3(perturbed))
    refusal = {"raised": False, "code": None, "detail": None}
    try:
        model(raw)
    except KeyAtomV3ContractError as exc:
        refusal = {"raised": True, "code": getattr(exc, "code", None),
                   "detail": str(exc)}
    forbidden = {"raised": False, "code": None, "detail": None}
    leaky = dict(fwd_clean)
    leaky["true_progress"] = torch.zeros(2)
    try:
        model(leaky)
    except KeyAtomV3ContractError as exc:
        forbidden = {"raised": True, "code": getattr(exc, "code", None),
                     "detail": str(exc)}
    return {
        "forward_view_identical": bool(view_identical),
        "outputs_identical": bool(outputs_identical),
        "supervision_keys": list(SUPERVISION_KEYS_V3),
        "supervision_key_refusal": refusal,
        "forbidden_metadata_refusal": forbidden,
        "passed": bool(view_identical and outputs_identical
                       and refusal["raised"] and forbidden["raised"]),
    }


def _finite_block(model: V541Model, env: FixtureEnv) -> Dict[str, Any]:
    """Unified loss forward+backward: every output and gradient finite."""
    positions = list(env.train_pos)
    lru = tr._FeatureLRU(env.ds, capacity=256, level=tr.FEATURE_LEVEL_V53MP)
    batch, records = tr._batch_from_positions(env.ds, lru, positions, "cpu")
    ep_weights = tr.estimate_endpoint_pos_weight(
        env.ds.samples, env.endpoint, positions, seed=AUDIT_SEED_V541)
    cfg = audit_config("E0" if model.is_e0 else model.embedding_id)
    set_training_stage_v541(model, backbone_trainable=True)
    model.train(True)
    model.zero_grad(set_to_none=True)
    loss_out = tr._loss_on_batch(
        model, batch, records, env.labels, env.endpoint, cfg,
        endpoint_pos_weight=float(ep_weights["pos_weight"]),
        pair_rng=tr._update_rng(AUDIT_SEED_V541, 1, salt=1))
    loss = loss_out["loss"]
    finite_loss = bool(torch.isfinite(loss).all())
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    missing = sum(1 for g in grads if g is None)
    nonfinite = sum(1 for g in grads
                    if g is not None and not bool(torch.isfinite(g).all()))
    with torch.no_grad():
        out = model(forward_view_v3(batch))
    outputs_finite = all(bool(torch.isfinite(v).all()) for v in out.values())
    model.zero_grad(set_to_none=True)
    return {
        "loss": float(loss.detach()),
        "loss_finite": finite_loss,
        "outputs_finite": bool(outputs_finite),
        "n_grad_tensors": len(grads),
        "n_grad_missing": int(missing),
        "n_grad_nonfinite": int(nonfinite),
        "passed": bool(finite_loss and outputs_finite and nonfinite == 0
                       and len(grads) > 0),
    }


def _degree_stats_block(env: FixtureEnv, *, legal: Mapping[str, Any],
                        work: Path) -> Dict[str, Any]:
    """E2: legal small inner-train fit + every forbidden path refused."""
    checks: List[Dict[str, Any]] = []
    checks.append(_check(
        "degree_stats_fitted_on_inner_train",
        legal.get("required") is True
        and legal.get("stats_sha256") is not None,
        result={k: v for k, v in legal.items() if k != "inputs"}))

    # the population must be EXACTLY the inner-train set — never dev/confirm
    if legal.get("path"):
        envelope = json.loads(Path(legal["path"]).read_text(encoding="utf-8"))
        pop = set(env.train_pos)
        forbidden = set(env.dev_pos) | set(env.confirm_pos)
        checks.append(_check(
            "degree_stats_population_is_inner_train_only",
            envelope.get("split") == DEGREE_STATS_SPLIT
            and not (pop & forbidden)
            and envelope.get("n_positions") == len(env.train_pos),
            split=envelope.get("split"),
            n_positions=envelope.get("n_positions"),
            expected=len(env.train_pos),
            leaked_positions=sorted(pop & forbidden)[:5]))
        checks.append(_check(
            "degree_stats_envelope_digest",
            degree_stats_payload_digest(envelope)
            == envelope.get("stats_sha256"),
            recomputed=degree_stats_payload_digest(envelope),
            recorded=envelope.get("stats_sha256")))
        checks.append(_check(
            "degree_stats_strict_loader_accepts",
            bool(load_degree_stats_v541(
                legal["path"], legal["sha256"])["mu_log"] is not None),
            path=legal["path"], sha256=legal["sha256"]))
        checks.append(_check(
            "degree_stats_fitted_outside_campaign_cache",
            str(Path(legal["path"]).resolve()).startswith(
                str(Path(work).resolve())),
            path=legal["path"], work=str(Path(work).resolve())))

    # forbidden split: fitting on dev/confirm is a TYPED refusal
    refusal_split = {"raised": False, "code": None}
    try:
        compute_degree_stats(env.ds.samples, list(env.dev_pos),
                             split="dev", seed=AUDIT_SEED_V541)
    except Exception as exc:                            # noqa: BLE001
        refusal_split = {"raised": True,
                         "code": getattr(exc, "code", None)
                         or type(exc).__name__}
    checks.append(_check(
        "degree_stats_forbidden_split_refused", refusal_split["raised"],
        **refusal_split))

    # a model must refuse to run WITHOUT its frozen statistics
    refusal_missing = {"raised": False, "code": None}
    try:
        build_audit_model("E2")
    except Exception as exc:                            # noqa: BLE001
        refusal_missing = {"raised": True,
                           "code": getattr(exc, "code", None)
                           or type(exc).__name__}
    checks.append(_check(
        "e2_refuses_missing_degree_stats", refusal_missing["raised"],
        **refusal_missing))

    # ... and a tampered envelope must be refused by the strict loader
    tamper = {"raised": False, "code": None}
    if legal.get("path"):
        bad_path = work / "degree_stats_tampered.json"
        doc = json.loads(Path(legal["path"]).read_text(encoding="utf-8"))
        doc["mu_log"] = float(doc["mu_log"]) + 1.0
        bad_path.write_text(json.dumps(doc, sort_keys=True),
                            encoding="utf-8")
        try:
            load_degree_stats_v541(str(bad_path), str(legal["sha256"]))
        except Exception as exc:                        # noqa: BLE001
            tamper = {"raised": True,
                      "code": getattr(exc, "code", None)
                      or type(exc).__name__}
    checks.append(_check("degree_stats_tampered_envelope_refused",
                         tamper["raised"], **tamper))

    return {"checks": checks, "passed": all(c["passed"] for c in checks),
            "fit": dict(legal)}


def audit_input_leak(env: FixtureEnv, *, work: Path,
                     degree_stats_for: Mapping[str, Any],
                     degree_fit: Mapping[str, Any]) -> Dict[str, Any]:
    """No-leak / whitelist / non-key action / finite / E2 / E6."""
    per_candidate: Dict[str, Any] = {}
    probe = make_v3_gate_batch(n_graphs=1)
    n_probe_keys = int(probe["key_index"].numel())
    n_probe_atoms = int(probe["pos"].shape[0])
    for eid in CANDIDATES_V541:
        model = build_audit_model(eid, degree_stats_for.get(eid))
        open_gate(model)
        intervention = _intervention_block(model)
        finite = _finite_block(model, env)
        probe_out = _call(model, forward_view_v3(probe))
        block = {
            "teacher_metadata_intervention": intervention,
            "finite_outputs_and_gradients": finite,
            "u_rows_equal_n_keys":
                tuple(probe_out["u"].shape)[0] == n_probe_keys,
            "n_atoms": n_probe_atoms,
            "n_keys": n_probe_keys,
            "node_vector_consumed": float(
                model.diagnostics.get("node_vector_consumed", 0.0)),
        }
        block["passed"] = bool(
            intervention["passed"] and finite["passed"]
            and block["u_rows_equal_n_keys"]
            and block["node_vector_consumed"] == 0.0)
        per_candidate[eid] = block

    e6 = {"raised": False, "code": None}
    try:
        V541Model(embedding_id="E6",
                  embedding_init_seed=AUDIT_INIT_SEED_V541)
    except Exception as exc:                            # noqa: BLE001
        e6 = {"raised": True,
              "code": getattr(exc, "code", None) or type(exc).__name__,
              "expected": "V541_EMBEDDING_NOT_IMPLEMENTED"}

    degree = _degree_stats_block(env, legal=degree_fit, work=work)
    passed = (all(c["passed"] for c in per_candidate.values())
              and e6["raised"] and degree["passed"])
    return {
        "schema": LEAK_SCHEMA,
        "generated_at": _utc_now(),
        "per_candidate": per_candidate,
        "e6_refused": e6,
        "degree_stats": degree,
        "passed": passed,
    }


# ---------------------------------------------------------------------------
# section 4 — restore: E0 20+20 vs 40 bitwise, E1-E5 2+2 vs 4
# ---------------------------------------------------------------------------


def _strip_timing(row: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in row.items()
            if k not in ("elapsed_s", "phase_timing", "resources")}


def _log_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(ln) for ln in
            path.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _load(path: Path) -> Dict[str, Any]:
    return torch.load(str(path), map_location="cpu", weights_only=False)


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v)
                for k, v in sorted(obj.items(), key=lambda kv: str(kv[0]))}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        arr = obj.detach().cpu().contiguous()
        try:
            raw = arr.numpy().tobytes()
        except (TypeError, RuntimeError):
            raw = arr.view(torch.uint8).numpy().tobytes()
        return {"dtype": str(arr.dtype), "shape": list(arr.shape),
                "sha256": hashlib.sha256(raw).hexdigest()}
    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        return {"dtype": str(arr.dtype), "shape": list(arr.shape),
                "sha256": hashlib.sha256(arr.tobytes()).hexdigest()}
    if hasattr(obj, "item") and getattr(obj, "ndim", 1) == 0:
        return obj.item()
    return obj


def _optimizer_digest(payload: Mapping[str, Any]) -> str:
    return _sha(_jsonable(payload["optimizer"]))


def h0_verdict(items: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Fail-closed roll-up: H0 is granted ONLY when EVERY item passed.

    §11.1 H0 ("schema/no-leak/confirm 隔离/restore/E(3)/permutation/
    非关键原子动作 全过"): one missing/failed item withholds the gate —
    there is no partial grant and no ordering-dependent short-circuit.
    """
    passed = all(bool(i.get("passed")) for i in items)
    return {
        "passed": passed,
        "n_items": len(items),
        "n_passed": sum(1 for i in items if i.get("passed")),
        "failed_items": [i.get("check") for i in items
                         if not i.get("passed")],
        "decision": ("H0 granted — every legality item passed" if passed
                     else "H0 NOT granted — fail-closed (§11.1 H0: "
                          "全过才通过)"),
    }


def _restore_one(embedding_id: str, env: FixtureEnv, *,
                 n1: int, n2: int,
                 degree_stats: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """``n1 + n2`` resumed vs ``n1 + n2`` uninterrupted, same fixture."""
    total = n1 + n2
    resumed_dir = Path(tempfile.mkdtemp(
        prefix="h0_resume_{}_".format(embedding_id)))
    fresh_dir = Path(tempfile.mkdtemp(
        prefix="h0_fresh_{}_".format(embedding_id)))
    try:
        def cfg(n: int) -> V541TrainerConfig:
            return audit_config(
                embedding_id, degree_stats=degree_stats,
                smoke_updates=int(n), freeze_encoder_updates=1,
                joint_updates=max(int(total) - 1, 1),
                val_every=1, grad_monitor_every=1)

        def run(c: V541TrainerConfig, out: Path, *, resume: bool) -> Dict:
            # ``train_positions`` is the FULL pool (train + inner-val
            # curves) so the trainer never re-derives a pool that would
            # sweep in the confirm fold — the §7.1 firewall refuses that,
            # and H0 proves the caller-side discipline instead.
            return train_run_v541(env.ds, env.labels, env.endpoint, c, out,
                                  dev_positions=env.dev_pos,
                                  train_positions=env.pool_pos,
                                  resume=resume)

        leg1 = run(cfg(n1), resumed_dir, resume=False)
        manifest_leg1 = (resumed_dir / "manifest.jsonl").read_text(
            encoding="utf-8")
        leg2 = run(cfg(total), resumed_dir, resume=True)
        manifest_leg2 = (resumed_dir / "manifest.jsonl").read_text(
            encoding="utf-8")
        fresh = run(cfg(total), fresh_dir, resume=False)
        # leg 1 must have stopped exactly where leg 2 resumed FROM
        leg1_ok = (int(leg1["schedule"]["updates_run"]) == n1
                   and int(leg1["selected_update"]) <= n1)
        # §7.3/§12.2: a resume leg APPENDS — leg 1's manifest lines must
        # survive byte-for-byte (never rewritten, never truncated).
        append_ok = (len(manifest_leg2) > len(manifest_leg1)
                     and manifest_leg2.startswith(manifest_leg1))

        a = _load(resumed_dir / "checkpoint_last.pt")
        b = _load(fresh_dir / "checkpoint_last.pt")
        model_equal = bool(
            set(a["model"]) == set(b["model"])
            and all(torch.equal(a["model"][k], b["model"][k])
                    for k in a["model"]))
        schedule_equal = a["schedule"] == b["schedule"]
        digest_equal = (a["content_digest_sha256"]
                        == b["content_digest_sha256"])
        opt_equal = _optimizer_digest(a) == _optimizer_digest(b)
        ba = _load(resumed_dir / "checkpoint_best.pt")
        bb = _load(fresh_dir / "checkpoint_best.pt")
        best_equal = (ba["content_digest_sha256"]
                      == bb["content_digest_sha256"])

        rows_r = {e["update"]: e for e in
                  _log_rows(resumed_dir / "train_log.jsonl")
                  if e.get("event") == "train"}
        rows_f = {e["update"]: e for e in
                  _log_rows(fresh_dir / "train_log.jsonl")
                  if e.get("event") == "train"}
        log_equal = bool(
            set(rows_r) == set(rows_f)
            and all(_strip_timing(rows_r[u]) == _strip_timing(rows_f[u])
                    for u in rows_f))
        val_r = {e["update"]: e["inner_val_loss"]
                 for e in _log_rows(resumed_dir / "train_log.jsonl")
                 if e.get("event") == "inner_val"}
        val_f = {e["update"]: e["inner_val_loss"]
                 for e in _log_rows(fresh_dir / "train_log.jsonl")
                 if e.get("event") == "inner_val"}

        man_r = read_manifest(resumed_dir / "manifest.jsonl")
        man_f = read_manifest(fresh_dir / "manifest.jsonl")
        resume_events = [e for e in man_r if e.get("event") == "resume"]
        run_complete = [e for e in man_r if e.get("event") == "run_complete"]

        passed = all((leg1_ok, append_ok, model_equal, schedule_equal,
                      digest_equal, opt_equal, best_equal, log_equal,
                      val_r == val_f,
                      leg2["sampler"]["batch_schedule_sha256"]
                      == fresh["sampler"]["batch_schedule_sha256"],
                      a["update"] == total, len(resume_events) == 1,
                      # one completed run per leg, APPENDED (never re-run
                      # over the earlier leg's line)
                      len(run_complete) == 2, bool(man_f)))
        return {
            "embedding_id": embedding_id,
            "legs": "{}+{} vs {}".format(n1, n2, total),
            "passed": passed,
            "leg1_stopped_where_leg2_resumed": leg1_ok,
            "leg1_selected_update": leg1["selected_update"],
            "manifest_appended_not_overwritten": append_ok,
            "manifest_entries_resumed": len(man_r),
            "manifest_entries_fresh": len(man_f),
            "manifest_run_complete_events": len(run_complete),
            "model_bitwise": model_equal,
            "optimizer_state": opt_equal,
            "best_checkpoint_digest": best_equal,
            "last_content_digest": digest_equal,
            "schedule_chain": schedule_equal,
            "schedule_chain_sha256": a["schedule"]["chain_sha256"],
            "train_log_semantic_rows": log_equal,
            "inner_val_by_update": val_r == val_f,
            "manifest_events": [e.get("event") for e in man_r],
            "resume_event": resume_events[0] if resume_events else None,
            "selected_update": leg2["selected_update"],
            "fresh_selected_update": fresh["selected_update"],
            "last_train_loss_resumed": leg2["last_train_loss"],
            "last_train_loss_fresh": fresh["last_train_loss"],
        }
    finally:
        shutil.rmtree(resumed_dir, ignore_errors=True)
        shutil.rmtree(fresh_dir, ignore_errors=True)


def _tamper_refusals(embedding_id: str, env: FixtureEnv, *,
                     degree_stats: Optional[Mapping[str, Any]]
                     ) -> Dict[str, Any]:
    """Identity + content-digest tampering must type-refuse (§7.3)."""
    out = Path(tempfile.mkdtemp(prefix="h0_tamper_"))
    try:
        def cfg(seed: int, n: int, **over: Any) -> V541TrainerConfig:
            return audit_config(
                embedding_id, degree_stats=degree_stats, seed=seed,
                smoke_updates=int(n), freeze_encoder_updates=1,
                val_every=1, grad_monitor_every=1, **over)

        def run(c: V541TrainerConfig, *, resume: bool) -> None:
            train_run_v541(env.ds, env.labels, env.endpoint, c, out,
                           dev_positions=env.dev_pos,
                           train_positions=env.pool_pos, resume=resume)

        run(cfg(AUDIT_SEED_V541, 2, joint_updates=1), resume=False)
        path = out / "checkpoint_last.pt"

        # (a) identity tamper: a DIFFERENT run seed must be refused
        seed_refusal = {"raised": False, "code": None}
        try:
            run(cfg(AUDIT_SEED_V541 + 1, 4, joint_updates=3), resume=True)
        except V541TrainerError as exc:
            seed_refusal = {"raised": True, "code": exc.code,
                            "fields": list(getattr(exc, "fields", []))}
        manifest_before = (out / "manifest.jsonl").read_text(encoding="utf-8")

        # (b) content-digest tamper: edit a weight, keep the stale digest
        payload = _load(path)
        key = next(iter(payload["model"]))
        payload["model"][key] = payload["model"][key] + 1.0
        torch.save(payload, str(path))
        digest_refusal = {"raised": False, "code": None}
        try:
            run(cfg(AUDIT_SEED_V541, 4, joint_updates=3), resume=True)
        except V541TrainerError as exc:
            digest_refusal = {"raised": True, "code": exc.code,
                              "fields": list(getattr(exc, "fields", []))}
        manifest_after = (out / "manifest.jsonl").read_text(encoding="utf-8")
        return {
            "identity_tamper": seed_refusal,
            "content_digest_tamper": digest_refusal,
            "refusals_did_not_append_resume":
                manifest_after == manifest_before,
            "passed": bool(
                seed_refusal["raised"]
                and seed_refusal["code"] == "RESUME_IDENTITY_MISMATCH"
                and digest_refusal["raised"]
                and digest_refusal["code"] == "RESUME_CONTENT_DIGEST"
                and manifest_after == manifest_before),
        }
    finally:
        shutil.rmtree(out, ignore_errors=True)


def audit_resume(env: FixtureEnv, *,
                 degree_stats_for: Mapping[str, Any]) -> Dict[str, Any]:
    """§7.3 restore equivalence + typed refusals, per candidate."""
    per_candidate: Dict[str, Any] = {}
    for eid in CANDIDATES_V541:
        stats = degree_stats_for.get(eid)
        # E0 is the parity anchor: the full 20+20 vs 40 bitwise protocol
        # (§7.3 test 1).  E1-E5 get the bounded 2+2 vs 4 form.
        n1 = 20 if eid == "E0" else 2
        block = _restore_one(eid, env, n1=n1, n2=n1, degree_stats=stats)
        block["tamper"] = _tamper_refusals(eid, env, degree_stats=stats)
        block["passed"] = bool(block["passed"] and block["tamper"]["passed"])
        per_candidate[eid] = block
    return {
        "schema": RESUME_SCHEMA,
        "generated_at": _utc_now(),
        "protocol": {
            "E0": "20+20 vs 40 bitwise (model/optimizer/schedule/digest/"
                  "log/inner-val)",
            "E1-E5": "2+2 vs 4 same path/SHA/manifest/digest + schedule "
                     "continuity + identity & digest typed refusal",
            "note": ("a live-process kill/resume belongs to the 200-update "
                     "smoke (§7.3 test 2), not to this fixture audit"),
        },
        "per_candidate": per_candidate,
        "passed": all(c["passed"] for c in per_candidate.values()),
    }


# ---------------------------------------------------------------------------
# assembly + CLI
# ---------------------------------------------------------------------------


def run_h0_legality_audit(*, root: Path, config_path: Optional[Path],
                          force: bool = False) -> Dict[str, Any]:
    """Run every H0 section and write the four artifacts."""
    root = Path(root)
    audits = root / "audits"
    audits.mkdir(parents=True, exist_ok=True)
    targets = [audits / n for n in ("h0_legality.json",
                                    "geometry_invariance.json",
                                    "input_leak.json",
                                    "resume_equivalence.json")]
    existing = [p for p in targets if p.is_file()]
    if existing and not force:
        raise H0AuditError(
            "H0_OUTPUT_EXISTS",
            "{} already exist; pass --force to overwrite".format(
                ", ".join(str(p) for p in existing)))

    work = audits / "h0_work"
    if work.exists():
        shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True, exist_ok=True)
    git = git_state().get("commit")
    started = time.time()

    # --- fixture (synthetic PD-1; pool + dev + confirm folds) --------------
    env = build_fixture_env(work / "tables")
    cfg = load_config(config_path)

    # --- E2 degree stats: ONE legal inner-train fit, cached under the work
    # --- dir (never the real campaign cache, never dev/confirm) -----------
    degree_fit = _fit_degree_stats(
        cfg, env, cache_dir=work / "degree_stats_cache", root=root, git=git)
    degree_stats_for: Dict[str, Any] = {}
    if degree_fit.get("required"):
        degree_stats_for["E2"] = degree_fit

    sections: Dict[str, Any] = {}
    for name, fn in (
            ("schema_gates", lambda: audit_schema_and_gates(
                root=root, config_path=config_path)),
            ("geometry_invariance", lambda: audit_geometry(
                degree_stats_for=degree_stats_for)),
            ("input_leak", lambda: audit_input_leak(
                env, work=work, degree_stats_for=degree_stats_for,
                degree_fit=degree_fit)),
            ("resume_equivalence", lambda: audit_resume(
                env, degree_stats_for=degree_stats_for))):
        try:
            sections[name] = fn()
            sections[name]["error"] = None
        except Exception as exc:                        # noqa: BLE001
            sections[name] = {
                "schema": None, "passed": False,
                "error": "{}: {}".format(type(exc).__name__, exc),
            }

    # --- write the three section artifacts ---------------------------------
    geo = sections["geometry_invariance"]
    leak = sections["input_leak"]
    resume = sections["resume_equivalence"]
    for path, doc in ((audits / "geometry_invariance.json", geo),
                      (audits / "input_leak.json", leak),
                      (audits / "resume_equivalence.json", resume)):
        path.write_text(json.dumps(doc, indent=1, sort_keys=True,
                                   default=str) + "\n", encoding="utf-8")

    # --- the H0 verdict ----------------------------------------------------
    items: List[Dict[str, Any]] = list(
        sections["schema_gates"].get("items") or [])
    for key, label, doc in (
            ("geometry_invariance",
             "E(3) + permutation + scalars + batch consistency at "
             "alpha={}".format(AUDIT_ALPHA_V541), geo),
            ("input_leak",
             "no-leak / whitelist / non-key action / finite / E2 stats / "
             "E6 refusal", leak),
            ("resume_equivalence",
             "restore equivalence + typed refusals", resume)):
        path = audits / "{}.json".format(key)
        items.append(_check(
            key, bool(doc.get("passed")), label=label, path=str(path),
            sha256=(sha256_file(path) if path.is_file() else None),
            error=doc.get("error")))

    for cert_name, label in (
            ("v53_parity.json", "E0 CPU strict-load + forward parity"),
            ("v53_parity_gpu.json", "E0 GPU forward parity")):
        cert = audits / cert_name
        if cert.is_file():
            doc = json.loads(cert.read_text(encoding="utf-8"))
            items.append(_check(
                "e0_parity_certificate_{}".format(
                    cert_name.rsplit(".", 1)[0]),
                doc.get("passed") is True, label=label, path=str(cert),
                sha256=sha256_file(cert), certificate_passed=doc.get("passed"),
                embedding_id=doc.get("embedding_id"),
                entries=len(doc.get("entries") or [])))
        else:
            items.append(_check(
                "e0_parity_certificate_{}".format(
                    cert_name.rsplit(".", 1)[0]), False, label=label,
                path=str(cert), reason="certificate missing (fail-closed)"))

    rollup = h0_verdict(items)
    verdict = {
        "schema": H0_LEGITIMACY_SCHEMA,
        "gate": "H0",
        "campaign_id": cfg.get("campaign_id"),
        "objective": cfg.get("objective"),
        "generated_at": _utc_now(),
        "generated_by": "scripts/campaigns/v541_legality_audit.py",
        "level": "fixture_restore",
        "git_commit": git,
        "cost": {
            "wall_seconds": round(time.time() - started, 3),
            "max_rss_bytes": host_peak_rss_bytes(),
            "device": "cpu",
            "new_qc_calls": 0,
            "real_50k_training": False,
            "dev_read": False,
            "confirm_read": False,
        },
        "fixture": {
            "tables_dir": str(env.tables_dir),
            "tables_sha256": {p.name: sha256_file(p)
                              for p in sorted(env.tables_dir.glob("*.csv"))},
            "folds": dict(FIXTURE_FOLDS_V541),
            "n_train_positions": len(env.train_pos),
            "n_inner_val_positions": len(env.inner_val_pos),
            "n_dev_positions": len(env.dev_pos),
            "n_confirm_positions": len(env.confirm_pos),
            "train_positions_sha256": positions_sha256(env.train_pos),
            "dev_curve_ids": env.dev_curve_ids,
            "confirm_curve_ids": env.confirm_curve_ids,
        },
        "degree_stats": degree_stats_for,
        "sections": {k: {"passed": bool(v.get("passed")),
                         "error": v.get("error")}
                     for k, v in sections.items()},
        "items": items,
        "n_items": rollup["n_items"],
        "n_passed": rollup["n_passed"],
        "failed_items": rollup["failed_items"],
        "passed": rollup["passed"],
        "decision": rollup["decision"],
    }
    (audits / "h0_legality.json").write_text(
        json.dumps(verdict, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    return verdict


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", default=DEFAULT_ROOT)
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        verdict = run_h0_legality_audit(
            root=Path(args.root),
            config_path=Path(args.config) if args.config else None,
            force=bool(args.force))
    except H0AuditError as exc:
        print("{}: {}".format(exc.code, exc), file=sys.stderr)
        return 2
    print(json.dumps({
        "passed": verdict["passed"],
        "n_items": verdict["n_items"],
        "n_passed": verdict["n_passed"],
        "decision": verdict["decision"],
        "wall_seconds": verdict["cost"]["wall_seconds"],
        "artifacts": str(Path(args.root) / "audits"),
    }, indent=2, sort_keys=True))
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
