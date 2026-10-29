#!/usr/bin/env python
"""V5.4.1 WP1b — historical baseline repair re-evaluation (plan §4.3).

WHAT THIS IS
------------
A bounded, fail-closed **re-evaluation** (never a retrain) of the frozen
V5.3-MP 50K assets through the WP1a-pair-repaired evaluation pipeline:

* historical full-task **C × seeds 17/29/43** — full metric set on each
  seed's rebuildable curve-level inner-val (8,192-sample cap) AND the
  frozen dev split;
* the frozen **R3 reference** (H2 MOVE guard baseline) — MOVE metrics
  only, on the same per-seed inner-val and the same dev split;
* per checkpoint × split the model runs **ONE prediction pass**; from that
  SINGLE prediction + the same labels both metric variants are built:
  ``before`` = audit-only LEGACY key-ptr g53 (the pre-WP1a algorithm,
  replicated verbatim below — it must never flow back into production)
  and ``after`` = the WP1a shared node-ptr g53
  (:func:`pes2ts_core.pesn.offline.evaluation.pair_ownership_v3.assign_batch_pairs`,
  the same object both production entry points import);
* q / ordering / endpoint metrics are computed ONCE from the same raw
  prediction (they never see pairs) — before/after identical by
  construction, and fidelity-gated on dev against historical
  ``metrics.json`` (same forward composition as the historical q pass);
* historical metrics are NEVER the "before" number — they are an
  independent fidelity anchor for the legacy replica only.

SPLITS & OUTPUTS (plan §4.3 / §9.3)
------------------------------------
* ``audits/evaluator_before_after.json``  — inner-val before/after
  (READABLE) + asset freeze + dev SEALED status (pass/fail flags +
  denominator COUNTS only — no dev science values).
* ``audits/dev_baseline_sealed/**``       — detailed dev results, SEALED:
  no unseal / no use for candidate design before WP7; WP7 candidate
  evaluation is a SEPARATE registered access.
* ``audits/dev_access_ledger.jsonl``      — append-only access ledger
  (actor, purpose=baseline_repair, curve/position SHA, code SHA, time,
  output SHA).
* ``audits/smoke_report.json`` / ``verify_report.json`` /
  ``failure_*.json``                      — consistency / integrity records.

FIDELITY NOTE (forward composition)
-----------------------------------
The single required prediction pass runs over ALL split positions
(chunk 64 — byte-identical to the historical q/endpoint pass).  The
historical MOVE pass ran over the ok-only subset (different GEMM shapes →
possible float32 drift ~1e-9).  A forward-drift probe (smoke stage, two
compositions on inner-val) measures the u/q noise and derives the
dead-zone count tolerance; u-independent pair-set gates (n_sign/n_active
pairs) stay EXACT, and sign/SRE outcomes are gated only when the probe
shows composition invariance (otherwise recorded inside the sealed files).

ORDERING TOLERANCE (derived, never result-tuned)
------------------------------------------------
Pairwise ordering accuracy is a ranking statistic over ~7.8M float32
comparisons: a comparison ``qp[later] > qp[earlier]`` flips between runs
only when the two q values sit within their combined float32 uncertainty.
The tolerance is therefore computed from INDEPENDENT inputs — the
inner-val measured q-noise band plus this run's own q ULPs
(``_ordering_tie_tols``) — never from the historical diff.  Companion
gates stay frozen/exact (n_pairs, n_eval_curves exact; q_mae/q_corr at
the paired-CI tolerances), so a structural prediction divergence is still
caught even if the tie bound is loose.

HARD BOUNDARIES
---------------
* read-only on every historical artifact (checkpoints, metrics, run
  records, config, split, reports);
* NO confirm access (the pool split's typed refusal is the guard);
* NO training, NO QC calls (new_QC_calls = 0), NO E1–E6 predictions;
* stdout / final replies expose only pass/fail + denominator COUNTS;
* real execution under ``scripts/infra/cgroup_scope.sh 12G 10G`` (single
  lane, swap0), columnar/dev-pack low-memory paths only — never a v1 full
  dataset/label cache;
* any asset / certificate / denominator that fails to close → fail-closed
  record + stop (never skip the R3 guard).

Stages: ``smoke`` (fixtures + one inner-val position vs BOTH production
entry points + forward-drift probe) → ``run --splits inner-val`` (readable)
→ ``run --splits dev`` (sealed + ledger) → ``verify``.
"""
from __future__ import annotations

import argparse
import fcntl
import getpass
import hashlib
import json
import os
import resource
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# numpy-only shared tool (the WP1a fix) — module-level, light.
from pes2ts_core.pesn.offline.evaluation.pair_ownership_v3 import (  # noqa: E402
    assign_batch_pairs,
    merge_pair_audits,
)
# light (no torch at module import; torch/eval imports stay function-local).
from scripts.campaigns.r3_v53mp_scale_paired_ci import (  # noqa: E402
    TIER_DEFAULTS,
    _build_tier,
    _collect,
    _ok_mask,
    _trainer_config,
)

# ---------------------------------------------------------------------------
# frozen identities / paths
# ---------------------------------------------------------------------------
SCHEMA_INNER_VAL = "v541_evaluator_before_after_v1"
SCHEMA_SEALED = "v541_dev_baseline_sealed_v1"
SCHEMA_SEAL_MANIFEST = "v541_dev_seal_manifest_v1"
SCHEMA_LEDGER = "v541_dev_access_ledger_v1"
SCHEMA_SMOKE = "v541_baseline_smoke_v1"
SCHEMA_VERIFY = "v541_baseline_verify_v1"
SCHEMA_FAILURE = "v541_baseline_failure_v1"

TIER = "50k"
SEEDS_DEFAULT = (17, 29, 43)
R3_RUN_ID = "5372d9b12807c3f0"
R3_CKPT = Path("outputs/pesn/30h_campaign/r3_runs") / R3_RUN_ID / "checkpoint_best.pt"
R3_METRICS = R3_CKPT.parent / "metrics.json"
C_RUN_ROOT = Path("outputs/pesn/v53mp/s4_50k")
CONFIG_FILE = Path("config/campaigns/pes2ts_v53mp_50k_20261026.yaml")
RUN_RECORDS = C_RUN_ROOT / "run_records.jsonl"

AUDIT_DIR = Path("outputs/pesn/v541_embed_50k/audits")
INNER_VAL_JSON = AUDIT_DIR / "evaluator_before_after.json"
SEALED_DIR = AUDIT_DIR / "dev_baseline_sealed"
LEDGER_PATH = AUDIT_DIR / "dev_access_ledger.jsonl"
SMOKE_REPORT = AUDIT_DIR / "smoke_report.json"
VERIFY_REPORT = AUDIT_DIR / "verify_report.json"
CGROUP_DIR = AUDIT_DIR / "cgroup"
GPU_LOCK = "/tmp/opencode/pes2ts_gpu.lock"

MOVE_BATCH = 64                    # mirrors _predict_batches / evaluate_v53mp
INNER_VAL_CAP_DEFAULT = 8192
INNER_VAL_CAP_SALT = 0x1E7A1       # trainer's seeded cap salt (key_atom_v53mp)
H2_RMSE_GUARD_PCT = 2.0            # plan H2 (existing guards formula)
PROBE_POSITIONS = 128
PROBE_NOISE_SANITY_MAX = 1.0e-4    # u drift above this = broken reproducibility

# fidelity tolerances vs historical metrics.json (dev).  Where a precedent
# exists (frozen paired-CI faithfulness gate) the SAME number is reused;
# cross-forward-composition floats get a documented wider band.
TOL = {
    # same forward composition as the historical q/endpoint pass (chunk 64
    # over ALL dev positions) → paired-CI precedent numbers apply:
    "q_mae_curve_mean": 1e-6,
    "q_corr_curve_mean": 1e-6,
    "endpoint_auprc": 1e-3,
    "endpoint_auroc": 1e-6,
    "endpoint_prevalence": 1e-6,
    # different forward composition (all-positions pass vs historical
    # move-only pass) → float32 GEMM-shape drift band; still 3-4 orders
    # above the measured cross-process noise (1.8e-9) and far below any
    # structural error (wrong model/labels/split):
    "median_rmse_over_zero": 1e-4,
    "cos_eff_median": 1e-4,
    # legacy pair outcomes: gated only when the drift probe proves
    # composition invariance (see _dev_fidelity_gates):
    "legacy_sign_median": 1e-9,
    "legacy_sre_median": 1e-6,
    # NOTE: ordering accuracy has NO frozen tolerance (the paired-CI gate
    # never contained it). Its tolerance is DERIVED per run from the
    # inner-val q-noise probe + float32 ULPs — see _ordering_tie_tols.
}

#: files whose content hash is part of the code freeze (eval-path surface)
CODE_FILES = (
    "pes2ts_core/pesn/offline/evaluation/pair_ownership_v3.py",
    "pes2ts_core/pesn/offline/evaluation/key_atom_protocols.py",
    "pes2ts_core/pesn/offline/training/key_atom_bc_v3.py",
    "pes2ts_core/pesn/offline/training/key_atom_v53mp_train_v3.py",
    "pes2ts_core/pesn/offline/training/key_atom_v52_train_v3.py",
    "pes2ts_core/pesn/features/key_atom_graph_v3.py",
    "pes2ts_core/pesn/models/key_atom_heads_v3.py",
    "pes2ts_core/pesn/models/key_atom_egnn_v3.py",
    "pes2ts_core/pesn/offline/data/key_atom_dev_pack_v53mp.py",
    "pes2ts_core/pesn/offline/data/key_atom_dataset_v3.py",
    "pes2ts_core/pesn/offline/data/key_atom_dataset_v3_col.py",
    "pes2ts_core/pesn/contracts/key_atom_v3.py",
    "scripts/campaigns/r3_v53mp_scale_paired_ci.py",
    "scripts/campaigns/r3_v53mp_run.py",
    "scripts/campaigns/v541_baseline_repair.py",
)


class RepairError(RuntimeError):
    """Typed fail-closed refusal (missing asset / broken certificate)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__("%s: %s" % (code, detail))
        self.code = code


# ---------------------------------------------------------------------------
# small utilities (stdout discipline: counts + pass/fail only)
# ---------------------------------------------------------------------------


def _say(msg: str) -> None:
    print("[wp1b] %s" % msg, flush=True)


def _c_metrics(seed: int) -> Path:
    return C_RUN_ROOT / ("v53mp_50k_C_s%d" % seed) / "metrics.json"


def _c_ckpt(seed: int) -> Path:
    return C_RUN_ROOT / ("v53mp_50k_C_s%d" % seed) / "checkpoint_best.pt"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_positions_sha(positions: Sequence[int]) -> str:
    """sha256 over the JSON-canonicalised sorted position list."""
    return _sha256_text(json.dumps(sorted(int(p) for p in positions),
                                   separators=(",", ":")))


def _canonical_curves_sha(curves: Sequence[str]) -> str:
    return _sha256_text(json.dumps(sorted(str(c) for c in curves),
                                   separators=(",", ":")))


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    raise RepairError("JSON_TYPE", "cannot serialise %r" % (type(obj),))


def _write_json(path: Path, doc: Mapping[str, Any]) -> str:
    """Atomic write; returns the file's sha256."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(doc), indent=1, sort_keys=True,
                              ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    return _sha256_file(path)


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise RepairError("FILE_MISSING", str(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    out = subprocess.run(["git", *args], cwd=str(PROJECT_ROOT),
                         capture_output=True, text=True, check=True)
    return out.stdout


class _Cost:
    """Wall / CPU / RSS / GPU cost capture (no science values)."""

    def __init__(self) -> None:
        self.wall0 = time.monotonic()
        ru0 = resource.getrusage(resource.RUSAGE_SELF)
        self.cpu0 = ru0.ru_utime + ru0.ru_stime
        self.gpu_forward_s = 0.0
        self.forward_batches = 0
        self.positions_predicted = 0

    def note_forward(self, seconds: float, batches: int, positions: int) -> None:
        self.gpu_forward_s += float(seconds)
        self.forward_batches += int(batches)
        self.positions_predicted += int(positions)

    def snapshot(self, device: str) -> Dict[str, Any]:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        gpu_peak = None
        if str(device).startswith("cuda"):
            import torch
            if torch.cuda.is_available():
                gpu_peak = int(torch.cuda.max_memory_allocated(device))
        return {
            "wall_s": round(time.monotonic() - self.wall0, 3),
            "cpu_s": round((ru.ru_utime + ru.ru_stime) - self.cpu0, 3),
            "peak_rss_mib": int(ru.ru_maxrss / 1024),
            "gpu_peak_bytes": gpu_peak,
            "gpu_forward_wall_s": round(self.gpu_forward_s, 3),
            "forward_batches": self.forward_batches,
            "positions_predicted": self.positions_predicted,
            "device": str(device),
            "new_QC_calls": 0,
        }


def _free_cuda() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class _GpuLock:
    """Single-lane GPU guard (same lock file the R3 driver uses)."""

    def __init__(self) -> None:
        self._fh: Any = None

    def acquire(self) -> None:
        Path(GPU_LOCK).parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(GPU_LOCK, "a+")
        try:
            fcntl.flock(self._fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._fh.close()
            self._fh = None
            raise RepairError(
                "GPU_LOCK_HELD",
                "%s is held by another lane; single-lane rule refuses" % GPU_LOCK)
        self._fh.seek(0)
        self._fh.truncate()
        self._fh.write("%s pid=%d\n" % (socket.gethostname(), os.getpid()))
        self._fh.flush()

    def release(self) -> None:
        if self._fh is not None:
            try:
                fcntl.flock(self._fh.fileno(), fcntl.LOCK_UN)
            finally:
                self._fh.close()
                self._fh = None


def _actor() -> str:
    try:
        user = getpass.getuser()
    except Exception:  # pragma: no cover
        user = "unknown"
    return "%s@%s#wp1b_baseline_repair" % (user, socket.gethostname())


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------------------
# legacy (pre-WP1a) g53 replica — AUDIT ONLY, never production
# ---------------------------------------------------------------------------
# Verbatim re-implementation of the pair-window filter REMOVED from
# key_atom_bc_v3.evaluate_checkpoint / r3_v53mp_scale_paired_ci._collect by
# WP1a: the KEY-row pointer (cumsum n_keys) windows batch-global NODE-row
# pairs.  Kept here so "before" can be reconstructed from the same
# prediction; production must never import this function again.


def _legacy_g53_samples(
    n_keys: np.ndarray,
    key_index: np.ndarray,
    pair_index: np.ndarray,
    pair_edit: np.ndarray,
    pos_all: np.ndarray,
    u_all: np.ndarray,
    y_all: np.ndarray,
    curve_ids: Sequence[str],
) -> List[Dict[str, Any]]:
    """Pre-WP1a g53 sample construction (key-ptr window on node rows)."""
    ptr = np.concatenate([[0], np.cumsum(n_keys)]).astype(np.int64)
    samples: List[Dict[str, Any]] = []
    for g in range(len(n_keys)):
        ks, ke = int(ptr[g]), int(ptr[g + 1])
        u_g = u_all[ks:ke]
        y_g = y_all[ks:ke]
        rows_g = key_index[ks:ke]
        pairs: List[Tuple[int, int]] = []
        edits: List[float] = []
        if pair_index.shape[1]:
            in_frame = np.nonzero(
                (pair_index[0] >= ptr[g]) & (pair_index[0] < ptr[g + 1]))[0]
            slot = {int(r): k for k, r in enumerate(rows_g)}
            for p in in_frame:
                a, b = int(pair_index[0, p]), int(pair_index[1, p])
                if a in slot and b in slot:
                    pairs.append((int(rows_g[slot[a]]), int(rows_g[slot[b]])))
                    edits.append(float(pair_edit[p]))
        samples.append({
            "pos": pos_all,
            "key_rows": [int(r) for r in rows_g],
            "pairs": pairs,
            "pair_edits": edits,
            "u": u_g.astype(np.float64),
            "y": y_g.astype(np.float64),
            "group_id": str(curve_ids[g]),
        })
    return samples


def _repaired_g53_samples(
    ownership: Any,
    pos_all: np.ndarray,
    u_all: np.ndarray,
    y_all: np.ndarray,
    curve_ids: Sequence[str],
    key_index: np.ndarray,
) -> List[Dict[str, Any]]:
    """WP1a g53 sample construction (shared node-ptr tool) — identical
    structure to both production entry points."""
    samples: List[Dict[str, Any]] = []
    for g, gs in enumerate(ownership.graphs):
        ks, ke = gs.key_lo, gs.key_hi
        u_g = u_all[ks:ke]
        y_g = y_all[ks:ke]
        rows_g = key_index[ks:ke]
        samples.append({
            "pos": pos_all,
            "key_rows": [int(r) for r in rows_g],
            "pairs": [(int(a), int(b)) for a, b in gs.pairs],
            "pair_edits": [float(e) for e in gs.pair_edits],
            "u": u_g.astype(np.float64),
            "y": y_g.astype(np.float64),
            "group_id": str(curve_ids[g]),
        })
    return samples


# ---------------------------------------------------------------------------
# per-position prediction record + single-pass collection
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class _PosPred:
    """ONE position's prediction + the structural arrays g53 needs.

    LOCAL rows only — group construction re-applies collate's node offsets,
    so g53 groups can be re-formed for any chunking (the legacy window is
    batch-composition dependent; MOVE groups must mirror the historical
    ok-only 64-chunks for a faithful "before").
    """

    position: int
    sample_id: str
    curve_id: str
    n_nodes: int
    n_keys: int
    pos: np.ndarray                 # [n,3] float32 (local)
    key_rows: np.ndarray            # [K] int64 (local node rows)
    pairs: np.ndarray               # [2,P] int64 (local node rows)
    edits: np.ndarray               # [P] float32
    u: np.ndarray                   # [K,3]
    y: np.ndarray                   # [K,3] y_teacher
    direction: str = ""
    q_pred: float = 0.0
    q_true: float = 0.0
    q_mask: bool = False
    ep_logit: float = 0.0
    ep_y01: float = 0.0
    ep_mask: bool = False
    ts_row: bool = False
    start_row: bool = False


def _predict_once(
    model: Any,
    positions: Sequence[int],
    cache: Any,
    device: str,
    *,
    collect_process: bool,
    labels: Any = None,
    endpoint: Any = None,
    batch_size: int = MOVE_BATCH,
) -> Tuple[Dict[int, _PosPred], Dict[str, Any]]:
    """ONE forward pass over ``positions`` (historical chunking, batch 64).

    Returns {position: _PosPred} + forward stats.  q/endpoint/label fields
    are filled only when ``collect_process`` (C models; R3 has no heads).
    """
    import torch

    from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3
    from pes2ts_core.pesn.features.key_atom_graph_v3 import collate_v3, to_torch_batch

    preds: Dict[int, _PosPred] = {}
    n_batches = 0
    model.eval()
    t0 = time.monotonic()
    with torch.no_grad():
        for start in range(0, len(positions), batch_size):
            chunk = [int(p) for p in positions[start:start + batch_size]]
            built = [cache[int(p)] for p in chunk]
            batch = to_torch_batch(collate_v3(built), device=device)
            out = model(forward_view_v3(batch))
            n_keys = batch["n_keys"].cpu().numpy().astype(np.int64)
            n_nodes = batch["n_nodes"].cpu().numpy().astype(np.int64)
            key_index = batch["key_index"].cpu().numpy().astype(np.int64)
            pair_index = batch["pair_index"].cpu().numpy().astype(np.int64)
            pair_edit = batch["pair_edit"].cpu().numpy()
            n_pairs = batch["n_pairs"].cpu().numpy().astype(np.int64)
            pos_all = batch["pos"].cpu().numpy()
            y_all = batch["y_teacher"].cpu().numpy()
            u_all = out["u"].detach().cpu().numpy()
            q_all = out["q"].detach().cpu().numpy() if collect_process else None
            ep_all = (out["endpoint_logit"].detach().cpu().numpy()
                      if collect_process else None)
            sample_ids = [str(s) for s in batch["sample_ids"]]
            curve_ids = [str(c) for c in batch["curve_ids"]]
            node_off = np.cumsum([0] + [int(v) for v in n_nodes]).astype(np.int64)
            key_ptr = np.cumsum([0] + [int(v) for v in n_keys]).astype(np.int64)
            pair_col = np.cumsum([0] + [int(v) for v in n_pairs]).astype(np.int64)
            for i in range(len(chunk)):
                k0, k1 = int(key_ptr[i]), int(key_ptr[i + 1])
                n0, n1 = int(node_off[i]), int(node_off[i + 1])
                c0, c1 = int(pair_col[i]), int(pair_col[i + 1])
                sid = sample_ids[i]
                rec = _PosPred(
                    position=chunk[i],
                    sample_id=sid,
                    curve_id=curve_ids[i],
                    n_nodes=n1 - n0,
                    n_keys=k1 - k0,
                    pos=pos_all[n0:n1],
                    key_rows=key_index[k0:k1] - n0,     # → local rows
                    pairs=pair_index[:, c0:c1] - n0,     # → local rows
                    edits=pair_edit[c0:c1].astype(np.float32),
                    u=u_all[k0:k1],
                    y=y_all[k0:k1],
                )
                if collect_process:
                    if labels is None or endpoint is None:
                        raise RepairError(
                            "PROCESS_INPUT_MISSING",
                            "collect_process requires labels+endpoint tables")
                    try:
                        lab = labels.by_sample[sid]
                    except KeyError as exc:
                        raise RepairError(
                            "LABEL_MISSING",
                            "sample %s has no label row" % sid) from exc
                    try:
                        epr = endpoint.by_sample[sid]
                    except KeyError as exc:
                        raise RepairError(
                            "ENDPOINT_LABEL_MISSING",
                            "sample %s has no endpoint row" % sid) from exc
                    rec.direction = str(lab.direction)
                    rec.q_true = 0.0 if lab.q is None else float(lab.q)
                    rec.q_mask = bool(lab.q_mask)
                    rec.q_pred = float(q_all[i])
                    rec.ep_logit = float(ep_all[i])
                    label = int(epr.endpoint_label)
                    rec.ep_y01 = 1.0 if label == 1 else 0.0
                    rec.ep_mask = label in (0, 1)
                    rec.ts_row = bool(epr.ts_row)
                    rec.start_row = bool(epr.start_row)
                preds[chunk[i]] = rec
            n_batches += 1
            if n_batches % 500 == 0:
                _say("predict batch %d/%d (positions %d/%d)" % (
                    n_batches, (len(positions) + batch_size - 1) // batch_size,
                    min(start + batch_size, len(positions)), len(positions)))
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    forward_s = time.monotonic() - t0
    model.train()
    stats = {"n_forward_batches": n_batches,
             "forward_wall_s": round(forward_s, 3),
             "n_positions": len(positions)}
    return preds, stats


# ---------------------------------------------------------------------------
# g53 group construction (mirrors collate_v3 offsets) + both variants
# ---------------------------------------------------------------------------


def _build_move_variants(
    preds: Mapping[int, _PosPred],
    move_positions: Sequence[int],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Build (before=legacy, after=repaired) g53 samples from ONE prediction.

    Groups ``move_positions`` in chunks of 64 — EXACTLY the historical
    evaluate_checkpoint chunking over the ok-only list (the legacy window
    filter is batch-composition dependent on it; the repaired window is
    not).  Global rows are rebuilt per group exactly like collate_v3.
    """
    legacy_all: List[Dict[str, Any]] = []
    repaired_all: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    n_input = n_legacy = n_repaired = 0
    for start in range(0, len(move_positions), MOVE_BATCH):
        group = [preds[int(p)] for p in move_positions[start:start + MOVE_BATCH]]
        offs = np.cumsum([0] + [r.n_nodes for r in group]).astype(np.int64)
        pos_all = np.concatenate([r.pos for r in group], axis=0)
        key_index = np.concatenate(
            [(r.key_rows + int(offs[i])).astype(np.int64)
             for i, r in enumerate(group)])
        pair_index = np.concatenate(
            [(r.pairs + int(offs[i])).astype(np.int64)
             for i, r in enumerate(group)], axis=1)
        pair_edit = np.concatenate([r.edits for r in group]).astype(np.float32)
        n_nodes = np.asarray([r.n_nodes for r in group], dtype=np.int64)
        n_keys = np.asarray([r.n_keys for r in group], dtype=np.int64)
        u_all = np.concatenate([r.u for r in group], axis=0)
        y_all = np.concatenate([r.y for r in group], axis=0)
        curve_ids = [r.curve_id for r in group]

        ownership = assign_batch_pairs(
            n_nodes=n_nodes, n_keys=n_keys, key_index=key_index,
            pair_index=pair_index, pair_edit=pair_edit)
        audits.append(ownership.audit(per_graph=False))
        legacy = _legacy_g53_samples(
            n_keys, key_index, pair_index, pair_edit,
            pos_all, u_all, y_all, curve_ids)
        repaired = _repaired_g53_samples(
            ownership, pos_all, u_all, y_all, curve_ids, key_index)
        legacy_all.extend(legacy)
        repaired_all.extend(repaired)
        n_input += int(pair_index.shape[1])
        n_legacy += sum(len(s["pairs"]) for s in legacy)
        n_repaired += sum(len(s["pairs"]) for s in repaired)
    meta = {
        "n_pairs_input": n_input,
        "n_pairs_collected_before": n_legacy,
        "n_pairs_collected_after": n_repaired,
        "pair_ownership_audit": merge_pair_audits(audits),
    }
    return legacy_all, repaired_all, meta


# ---------------------------------------------------------------------------
# metric blocks
# ---------------------------------------------------------------------------


def _move_metrics(
    legacy_samples: Sequence[Mapping[str, Any]],
    repaired_samples: Sequence[Mapping[str, Any]],
    pair_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    """before/after MOVE metrics from the SAME predictions.

    ``key_rmse_over_zero`` (RMSE/zero) and ``fixed_coverage_direction``
    (G5.2 effective-atom cosine) are PAIR-INDEPENDENT — computed once from
    the shared u/y and reported identically in both variants (structural
    guarantee: they never receive pairs).  Only g53 (sign / SRE / dead)
    and the pair denominators differ between before and after.
    """
    from pes2ts_core.pesn.offline.evaluation.key_atom_protocols import (
        evaluate_g52,
        evaluate_g53,
    )

    t0 = time.monotonic()
    u_list = [np.asarray(s["u"]) for s in repaired_samples]
    y_list = [np.asarray(s["y"]) for s in repaired_samples]
    curve_list = [str(s["group_id"]) for s in repaired_samples]
    g52 = evaluate_g52(u_list, y_list, curve_list, bootstrap_draws=0)
    rmse_block = g52["raw"]["aggregate"]
    n_key_atoms = int(sum(u.shape[0] for u in u_list))
    n_eff = int(rmse_block.get("n_effective_atoms") or 0)
    fixed_cov = {
        "cos_eff_median": rmse_block.get("cos_eff_median"),
        "n_effective_atoms": n_eff,
        "key_atom_coverage": (float(n_eff) / n_key_atoms) if n_key_atoms else None,
        "definition": ("G5.2 effective-atom (||y||>=0.02 A) soft-cosine "
                       "median; coverage fixed by the TEACHER, "
                       "pair-independent"),
    }
    g53_before = evaluate_g53(list(legacy_samples))
    g53_after = evaluate_g53(list(repaired_samples))
    t1 = time.monotonic()

    before = {
        "key_rmse_over_zero": rmse_block,
        "fixed_coverage_direction": fixed_cov,
        "sre_and_sign": g53_before["aggregate"],
    }
    after = {
        "key_rmse_over_zero": rmse_block,
        "fixed_coverage_direction": fixed_cov,
        "sre_and_sign": g53_after["aggregate"],
    }
    n_in = int(pair_meta["n_pairs_input"])
    n_before = int(pair_meta["n_pairs_collected_before"])
    n_after = int(pair_meta["n_pairs_collected_after"])
    pair_denoms = {
        "n_pairs_input": n_in,
        "n_pairs_collected_before": n_before,
        "n_pairs_collected_after": n_after,
        "n_pairs_dropped_before": n_in - n_before,
        "n_pairs_dropped_after": n_in - n_after,
        "pair_ownership_audit": pair_meta["pair_ownership_audit"],
    }
    delta = _move_delta(before, after, pair_denoms)
    return {
        "before_legacy": before,
        "after_repaired": after,
        "delta_after_minus_before": delta,
        "pair_denominators": pair_denoms,
        "pair_independent_fields": [
            "key_rmse_over_zero", "fixed_coverage_direction"],
        "metrics_wall_s": round(t1 - t0, 3),
    }


def _move_delta(before: Mapping[str, Any],
                after: Mapping[str, Any],
                pair_denoms: Mapping[str, Any]) -> Dict[str, Any]:
    b_s, a_s = before["sre_and_sign"], after["sre_and_sign"]
    b_r = before["key_rmse_over_zero"]
    a_r = after["key_rmse_over_zero"]

    def _d(key: str, bmap: Mapping[str, Any], amap: Mapping[str, Any]) -> Any:
        bv, av = bmap.get(key), amap.get(key)
        if bv is None or av is None:
            return None
        return float(av) - float(bv)

    return {
        "sign_accuracy_cluster_median": _d(
            "sign_accuracy_cluster_median", b_s, a_s),
        "sre_cluster_median": _d("sre_cluster_median", b_s, a_s),
        "dead_zone_fraction": _d("dead_zone_fraction", b_s, a_s),
        "n_sign_pairs": (int(a_s.get("n_sign_pairs") or 0)
                         - int(b_s.get("n_sign_pairs") or 0)),
        "n_active_pairs": (int(a_s.get("n_active_pairs") or 0)
                           - int(b_s.get("n_active_pairs") or 0)),
        "n_dead_pairs": (int(a_s.get("n_dead_pairs") or 0)
                         - int(b_s.get("n_dead_pairs") or 0)),
        "median_rmse_over_zero": _d("median_rmse_over_zero", b_r, a_r),
        "n_pairs_collected": (int(pair_denoms["n_pairs_collected_after"])
                              - int(pair_denoms["n_pairs_collected_before"])),
        "note": ("pair-independent fields (RMSE/zero, fixed-coverage "
                 "direction) are identical by construction — reported once"),
    }


def _process_metrics(preds: Sequence[_PosPred],
                     *, tau_end: Optional[float]) -> Dict[str, Any]:
    """q / ordering / endpoint from raw predictions — NEVER sees pairs.

    Mirrors evaluate_v53mp's q_head / ordering / endpoint blocks exactly
    (same grouping, same helpers) so dev fidelity vs metrics.json is a real
    prediction-path check, independent of the pair repair.
    """
    from pes2ts_core.pesn.offline.training.key_atom_v52_train_v3 import (
        auc_mann_whitney,
        pearson,
    )
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        _sigmoid,
        average_precision,
        ordering_pair_accuracy,
    )

    qp = np.asarray([p.q_pred for p in preds], dtype=np.float64)
    qt = np.asarray([p.q_true for p in preds], dtype=np.float64)
    qm = np.asarray([p.q_mask for p in preds], dtype=bool)
    groups = [(str(p.curve_id), str(p.direction)) for p in preds]

    per_curve: Dict[str, List[int]] = {}
    for i, (cid, _direction) in enumerate(groups):
        per_curve.setdefault(cid, []).append(i)
    q_curve_mae: List[float] = []
    q_curve_corr: List[float] = []
    for cid, idxs in sorted(per_curve.items()):
        ok = [i for i in idxs if qm[i]]
        if not ok:
            continue
        q_curve_mae.append(float(np.abs(qp[ok] - qt[ok]).mean()))
        corr = pearson(qp[ok], qt[ok])
        if corr is not None:
            q_curve_corr.append(corr)
    q_head = {
        "mae_curve_median": (float(np.median(q_curve_mae))
                             if q_curve_mae else None),
        "mae_curve_mean": (float(np.mean(q_curve_mae))
                           if q_curve_mae else None),
        "corr_curve_median": (float(np.median(q_curve_corr))
                              if q_curve_corr else None),
        "corr_curve_mean": (float(np.mean(q_curve_corr))
                            if q_curve_corr else None),
        "n_eval_curves": len(q_curve_mae),
    }

    valid_idx = [i for i in range(len(qp)) if qm[i]]
    full = ordering_pair_accuracy(
        [qp[i] for i in valid_idx], [qt[i] for i in valid_idx],
        [groups[i] for i in valid_idx], gap=0.0, cross_ts_only=False)
    cross = ordering_pair_accuracy(
        [qp[i] for i in valid_idx], [qt[i] for i in valid_idx],
        [groups[i] for i in valid_idx], gap=0.0, cross_ts_only=True)
    ordering = {
        "pairwise_accuracy_full": full["accuracy"],
        "n_pairs_full": full["n_pairs"],
        "pairwise_accuracy_cross_ts": cross["accuracy"],
        "n_pairs_cross_ts": cross["n_pairs"],
    }

    log = np.asarray([p.ep_logit for p in preds], dtype=np.float64)
    y = np.asarray([(p.ep_y01 if p.ep_mask else 0.0) for p in preds],
                   dtype=np.float64)
    m = np.asarray([p.ep_mask for p in preds], dtype=bool)
    keep = m & np.isin(y, (0, 1))
    scores = _sigmoid(log[keep])
    labels_keep = y[keep]
    auroc = auc_mann_whitney(scores.tolist(), labels_keep.tolist())
    auprc = average_precision(scores.tolist(), labels_keep.tolist())
    mis_start = None
    mis_ts = None
    n_start = int(np.sum(keep & np.asarray(
        [p.start_row for p in preds], dtype=bool)))
    n_ts = int(np.sum(keep & np.asarray(
        [p.ts_row for p in preds], dtype=bool)))
    if tau_end is not None:
        pred_stop = _sigmoid(log) >= float(tau_end)
        start_arr = np.asarray([p.start_row for p in preds], dtype=bool)
        ts_arr = np.asarray([p.ts_row for p in preds], dtype=bool)
        if n_start:
            mis_start = float(pred_stop[keep & start_arr].mean())
        if n_ts:
            mis_ts = float(pred_stop[keep & ts_arr].mean())
    endpoint = {
        "auroc": auroc,
        "auprc": auprc,
        "prevalence": (float(labels_keep.mean())
                       if int(labels_keep.size) else None),
        "n_unmasked": int(keep.sum()),
        "n_pos": int((labels_keep == 1).sum()),
        "n_neg": int((labels_keep == 0).sum()),
        "tau_end": tau_end,
        "start_misstop_rate_at_tau": mis_start,
        "ts_misstop_rate_at_tau": mis_ts,
        "n_start_rows": n_start,
        "n_ts_rows": n_ts,
    }
    return {"q_head": q_head, "ordering": ordering, "endpoint": endpoint}


def _h2_move_guard(c_move: Mapping[str, Any], r3_move: Mapping[str, Any],
                   *, split: str, variant: str, seed: Optional[int] = None,
                   ) -> Dict[str, Any]:
    """H2 RMSE guard components (existing guards formula, threshold 2%)."""

    def _blk(m: Mapping[str, Any]) -> Dict[str, Any]:
        agg = m["key_rmse_over_zero"]
        sgn = m["sre_and_sign"]
        return {
            "median_rmse": agg.get("median_rmse"),
            "median_rmse_over_zero": agg.get("median_rmse_over_zero"),
            "sign_accuracy_cluster_median":
                sgn.get("sign_accuracy_cluster_median"),
            "dead_zone_fraction": sgn.get("dead_zone_fraction"),
        }

    c, r3 = _blk(c_move), _blk(r3_move)

    def _pct(cv: Any, rv: Any) -> Optional[float]:
        if cv is None or rv is None or not float(rv):
            return None
        return 100.0 * (float(cv) - float(rv)) / float(rv)

    deg = _pct(c["median_rmse"], r3["median_rmse"])
    deg0 = _pct(c["median_rmse_over_zero"], r3["median_rmse_over_zero"])
    return {
        "split": split,
        "variant": variant,
        "c_seed": seed,
        "c_seed_move": c,
        "r3_reference_move": r3,
        "rmse_degradation_pct": deg,
        "within_rmse_guard_2pct": None if deg is None else deg <= H2_RMSE_GUARD_PCT,
        "rmse_over_zero_degradation_pct": deg0,
        "within_rmse0_guard_2pct": None if deg0 is None else deg0 <= H2_RMSE_GUARD_PCT,
        "formula": ("100*(c-r3)/r3 on median_rmse — mirrors evaluate_v53mp "
                    "guards (model_median_rmse vs r3_baseline_median_rmse)"),
        "threshold_pct": H2_RMSE_GUARD_PCT,
        "note": ("H2 comparison is the AFTER (repaired) variant; BEFORE is "
                 "reported for the audit only"),
    }


# ---------------------------------------------------------------------------
# gate helpers
# ---------------------------------------------------------------------------


def _compare(name: str, got: Any, want: Any, tol: float, mode: str) -> Dict[str, Any]:
    """mode: 'exact' (integer/None-paired) or 'abs' (float tolerance)."""
    if got is None and want is None:
        return {"name": name, "mode": mode, "tol": tol, "abs_diff": 0.0,
                "pass": True, "got": None, "want": None}
    if got is None or want is None:
        return {"name": name, "mode": mode, "tol": tol, "abs_diff": None,
                "pass": False, "got": _jsonable(got), "want": _jsonable(want)}
    if mode == "exact":
        if isinstance(got, str) or isinstance(want, str):
            ok = str(got) == str(want)
            diff = 0 if ok else 1
        else:
            ok = int(got) == int(want)
            diff = abs(int(got) - int(want))
    else:
        ok = abs(float(got) - float(want)) <= float(tol)
        diff = abs(float(got) - float(want))
    return {"name": name, "mode": mode, "tol": tol,
            "abs_diff": float(diff), "pass": bool(ok),
            "got": _jsonable(got), "want": _jsonable(want)}


def _record(name: str, got: Any, want: Any) -> Dict[str, Any]:
    """Non-gated fidelity record (values live ONLY in sealed/inner-val)."""
    g = _compare(name, got, want, 0.0, "abs")
    g["gated"] = False
    return g


def _pass_all(gates: Sequence[Mapping[str, Any]]) -> bool:
    return all(bool(g.get("pass")) for g in gates)


def _gate_flags(gates: Sequence[Mapping[str, Any]]) -> Dict[str, bool]:
    return {str(g["name"]): bool(g.get("pass")) for g in gates}


def _move_sanity(
    move: Mapping[str, Any],
    *,
    n_positions: int,
    n_positions_ok: int,
    n_curves_move: int,
) -> List[Dict[str, Any]]:
    """Structural denominator sanity (counts only — printable)."""
    den = move["pair_denominators"]
    audit = den["pair_ownership_audit"]
    b = move["before_legacy"]["sre_and_sign"]
    a = move["after_repaired"]["sre_and_sign"]
    n_in = int(den["n_pairs_input"])
    n_after = int(den["n_pairs_collected_after"])
    n_before = int(den["n_pairs_collected_before"])
    n_ignored = int(audit.get("n_ignored_non_key_total", 0))
    return [
        {"name": "s1_repaired_no_silent_pair_drop",
         "pass": (n_in - n_ignored - n_after) == 0,
         "detail": "input - ignored_non_key - collected_after == 0"},
        {"name": "s3_audit_invalid_zero",
         "pass": int(audit.get("n_invalid_pairs_total", 0)) == 0,
         "detail": "invalid pairs raise; success implies 0"},
        {"name": "s4_legacy_subset_of_repaired",
         "pass": n_before <= n_after,
         "detail": "collected_before <= collected_after"},
        {"name": "s5_sign_pairs_monotone",
         "pass": (int(a.get("n_sign_pairs") or 0)
                  >= int(b.get("n_sign_pairs") or 0)),
         "detail": "g53 n_sign_pairs after >= before"},
        {"name": "s6_active_pairs_monotone",
         "pass": (int(a.get("n_active_pairs") or 0)
                  >= int(b.get("n_active_pairs") or 0)),
         "detail": "g53 n_active_pairs after >= before"},
        {"name": "s7_denominators_nonzero",
         "pass": (n_positions > 0 and n_positions_ok > 0
                  and n_curves_move > 0 and n_in > 0),
         "detail": "positions/ok/curves/pairs all > 0"},
    ]


# ---------------------------------------------------------------------------
# asset freeze + integrity
# ---------------------------------------------------------------------------


def _freeze_assets() -> Dict[str, Any]:
    """Freeze SHA of every input WP1b depends on (fail-closed)."""
    tables_dir = Path(TIER_DEFAULTS[TIER]["tables_dir"])
    split_path = tables_dir / "split_manifest_v53mp.json"
    for p in (split_path, CONFIG_FILE, RUN_RECORDS, R3_CKPT, R3_METRICS):
        if not p.is_file():
            raise RepairError("ASSET_MISSING", str(p))

    head = _git("rev-parse", "HEAD").strip()
    diff_sha = _sha256_text(_git("diff", "HEAD"))
    porcelain_sha = _sha256_text(_git("status", "--porcelain"))
    code_files = {}
    for rel in CODE_FILES:
        p = PROJECT_ROOT / rel
        if not p.is_file():
            raise RepairError("CODE_FILE_MISSING", rel)
        code_files[rel] = _sha256_file(p)
    # identity = the eval-path files that actually execute — STRICTLY
    # compared between stages.  provenance (git head/diff/porcelain) is
    # recorded for the audit trail but NOT compared: the porcelain listing
    # includes unrelated working-tree files (concurrent sessions add
    # untracked docs/config), which are not part of this evaluation's
    # execution surface and would produce false drift.
    code_block = {
        "identity": code_files,
        "provenance": {
            "git_head": head,
            "git_diff_sha256": diff_sha,
            "git_porcelain_sha256": porcelain_sha,
        },
    }

    config_sha = _sha256_file(CONFIG_FILE)
    split_sha = _sha256_file(split_path)
    data_manifest_path = tables_dir / "data_manifest.json"
    if not data_manifest_path.is_file():
        raise RepairError("ASSET_MISSING", str(data_manifest_path))
    data_manifest_sha = _sha256_file(data_manifest_path)

    # run records for C×3 (read-only)
    records: Dict[int, Dict[str, Any]] = {}
    for line in RUN_RECORDS.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if (str(rec.get("arm")) == "C" and rec.get("tier") == TIER
                and rec.get("status") == "done"):
            seed = int(rec.get("seed", -1))
            records[seed] = rec
    for seed in SEEDS_DEFAULT:
        if seed not in records:
            raise RepairError(
                "RUN_RECORD_MISSING", "no done C run record for seed %d" % seed)

    seed_blocks: Dict[str, Any] = {}
    label_specs = set()
    for seed in SEEDS_DEFAULT:
        ckpt, mpath = _c_ckpt(seed), _c_metrics(seed)
        if not ckpt.is_file() or not mpath.is_file():
            raise RepairError("ASSET_MISSING", "%s / %s" % (ckpt, mpath))
        ckpt_sha = _sha256_file(ckpt)
        metrics = _read_json(mpath)
        prov = metrics.get("provenance") or {}
        rec = records[seed]
        checks = {
            "ckpt_eq_metrics_checkpoint_best_sha256":
                ckpt_sha == metrics.get("checkpoint_best_sha256"),
            "ckpt_eq_run_record_checkpoint_sha256":
                ckpt_sha == rec.get("checkpoint_sha256"),
            "config_sha_eq_provenance":
                config_sha == prov.get("config_sha256"),
            "config_sha_eq_run_record":
                config_sha == rec.get("config_sha256"),
            "split_sha_eq_provenance":
                split_sha == prov.get("split_sha256"),
            "split_sha_eq_run_record":
                split_sha == rec.get("split_sha256"),
            "data_manifest_sha_eq_provenance":
                data_manifest_sha == prov.get("data_manifest_sha256"),
            "schema_is_v53mp":
                metrics.get("schema") == "pesn_key_atom_v53mp_train_v1",
            "run_outcome_ok": metrics.get("run_outcome") == "ok",
        }
        bad = [k for k, v in checks.items() if not v]
        if bad:
            raise RepairError(
                "FREEZE_CROSSCHECK_FAILED",
                "seed %d failed: %s" % (seed, ", ".join(bad)))
        label_specs.add(str(prov.get("label_spec_sha256")))
        label_specs.add(str(prov.get("endpoint_label_spec_sha256")))
        seed_blocks[str(seed)] = {
            "run_id": "v53mp_50k_C_s%d" % seed,
            "checkpoint": {"path": str(ckpt), "sha256": ckpt_sha},
            "metrics": {"path": str(mpath), "sha256": _sha256_file(mpath)},
            "run_record": {
                "run_id": rec.get("run_id"),
                "sha256": _sha256_text(json.dumps(
                    rec, sort_keys=True, ensure_ascii=False)),
                "checkpoint_sha256": rec.get("checkpoint_sha256"),
                "config_sha256": rec.get("config_sha256"),
                "split_sha256": rec.get("split_sha256"),
            },
            "provenance": {
                "config_sha256": prov.get("config_sha256"),
                "split_sha256": prov.get("split_sha256"),
                "data_manifest_sha256": prov.get("data_manifest_sha256"),
                "label_spec_sha256": prov.get("label_spec_sha256"),
                "endpoint_label_spec_sha256":
                    prov.get("endpoint_label_spec_sha256"),
                "teacher_manifest_sha256": prov.get("teacher_manifest_sha256"),
                "features_sha256": prov.get("features_sha256"),
                "loader_version": prov.get("loader_version"),
                "loader_idx_sha256": prov.get("loader_idx_sha256"),
            },
            "dev_pack_provenance": prov.get("dev_pack"),
            "history_denominators": {
                "n_dev_samples": metrics.get("n_dev_samples"),
                "n_dev_curves": metrics.get("n_dev_curves"),
                "n_dev_samples_evaluated": metrics.get("n_dev_samples_evaluated"),
                "move_n_val_samples":
                    (metrics.get("move") or {}).get("n_val_samples"),
                "move_n_val_curves":
                    (metrics.get("move") or {}).get("n_val_curves"),
                "n_inner_val_samples": metrics.get("n_inner_val_samples"),
                "inner_val_curves": metrics.get("inner_val_curves"),
                "inner_val_max_samples": metrics.get("inner_val_max_samples"),
            },
            "history_move_metrics": {
                "key_rmse_over_zero":
                    (metrics.get("move") or {}).get("key_rmse_over_zero"),
                "sre_and_sign": (metrics.get("move") or {}).get("sre_and_sign"),
                "q_head": metrics.get("q_head"),
                "ordering": metrics.get("ordering"),
                "endpoint_head": metrics.get("endpoint_head"),
            },
        }
    if len(label_specs) != 1:
        raise RepairError(
            "LABEL_SPEC_DRIFT",
            "C seeds disagree on label spec SHAs: %s" % sorted(label_specs))

    r3_metrics = _read_json(R3_METRICS)
    r3_ckpt_sha = _sha256_file(R3_CKPT)
    r3_bad = [k for k, v in {
        "ckpt_eq_metrics_checkpoint_best_sha256":
            r3_ckpt_sha == r3_metrics.get("checkpoint_best_sha256"),
        "schema_is_key_atom_bc_v3":
            r3_metrics.get("schema") == "pesn_key_atom_bc_v3_train_v1",
        "arm_is_f2_bidir": r3_metrics.get("arm") == "F2-bidir",
    }.items() if not v]
    if r3_bad:
        raise RepairError("R3_FREEZE_FAILED", ", ".join(r3_bad))

    return {
        "schema": "v541_baseline_freeze_v1",
        "created_at": _now(),
        "code": code_block,
        "config": {"path": str(CONFIG_FILE), "sha256": config_sha},
        "split_manifest": {"path": str(split_path), "sha256": split_sha},
        "data_manifest": {"path": str(data_manifest_path),
                          "sha256": data_manifest_sha},
        "label_spec_sha256": sorted(label_specs)[0],
        "c_runs": seed_blocks,
        "r3_reference": {
            "run_id": R3_RUN_ID,
            "checkpoint": {"path": str(R3_CKPT), "sha256": r3_ckpt_sha},
            "metrics": {"path": str(R3_METRICS),
                        "sha256": _sha256_file(R3_METRICS)},
        },
        "tier": TIER,
        "read_only_inputs": True,
    }


def _code_identity_sha(freeze: Mapping[str, Any]) -> str:
    """Stable code SHA over the strictly-compared eval-path identity map."""
    return _sha256_text(json.dumps(
        (freeze.get("code") or {}).get("identity") or {},
        sort_keys=True, separators=(",", ":")))


def _assert_freeze_unchanged(recorded: Mapping[str, Any]) -> None:
    """Dev stage / verify: frozen inputs + eval-path code must not drift.

    Only ``code.identity`` (per-file sha256 of CODE_FILES) is compared;
    ``code.provenance`` (git head/diff/porcelain) is recorded for the
    audit trail but not compared — unrelated working-tree files outside
    the eval surface must not fail the freeze.
    """
    current = _freeze_assets()

    def _ident(block: Any) -> Mapping[str, Any]:
        block = block or {}
        return block.get("identity") or block.get("files") or {}

    cur_id = _ident(current.get("code"))
    rec_id = _ident(recorded.get("code"))
    if cur_id != rec_id:
        drifted = sorted(k for k in set(cur_id) | set(rec_id)
                         if cur_id.get(k) != rec_id.get(k))
        raise RepairError(
            "CODE_DRIFTED_SINCE_INNER_VAL",
            "eval-path files changed between stages: %s" % drifted)
    for key in ("config", "split_manifest", "data_manifest",
                "label_spec_sha256", "c_runs", "r3_reference"):
        if current.get(key) != recorded.get(key):
            raise RepairError(
                "FROZEN_ASSET_DRIFTED",
                "freeze block %r changed between stages" % key)


# ---------------------------------------------------------------------------
# corpus / split rebuild
# ---------------------------------------------------------------------------


def _open_corpus_light() -> Dict[str, Any]:
    """Columnar dataset + LAZY label tables + split doc (NO dev access)."""
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import LABEL_STATUSES
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (
        ColumnarKeyAtomDatasetV3,
    )

    tables_dir = Path(TIER_DEFAULTS[TIER]["tables_dir"])
    idx_dir = tables_dir / "idx_v2"
    if not idx_dir.is_dir():
        raise RepairError("IDX2_MISSING", str(idx_dir))
    split_path = tables_dir / "split_manifest_v53mp.json"
    dataset = ColumnarKeyAtomDatasetV3(
        tables_dir,
        split_manifest=str(split_path),
        label_statuses=LABEL_STATUSES,
        cache_curves=int(TIER_DEFAULTS[TIER]["cache_curves"]),
    )
    labels, endpoint = dataset.label_tables()
    split_doc = json.loads(split_path.read_text(encoding="utf-8"))
    return {"dataset": dataset, "labels": labels, "endpoint": endpoint,
            "split_doc": split_doc, "split_path": split_path,
            "tables_dir": tables_dir}


def _rebuild_inner_val(corpus: Mapping[str, Any], seed: int,
                       stored_cfg: Mapping[str, Any],
                       history: Mapping[str, Any],
                       ) -> Tuple[List[int], Dict[str, Any]]:
    """Rebuild the seed's curve-level inner-val + 8192 cap EXACTLY as the
    historical trainer did; fail-closed unless denominators match history."""
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        _curve_id_set_of,
        v53mp_pool_split,
    )
    from scripts.campaigns.r3_v53mp_run import _tier_pool_curve_ids

    frac = float(stored_cfg.get("inner_val_fraction", 0.2))
    cap = int(stored_cfg.get("inner_val_max_samples", INNER_VAL_CAP_DEFAULT))
    hist_cap = history.get("inner_val_max_samples")
    if hist_cap not in (None, cap):
        raise RepairError(
            "INNER_VAL_CAP_MISMATCH",
            "stored cap %s != history %s" % (cap, hist_cap))
    pool_curve_ids = _tier_pool_curve_ids(corpus["split_doc"], TIER)
    _train_pos, inner_val_pos = v53mp_pool_split(
        corpus["dataset"].samples, pool_curve_ids=pool_curve_ids,
        seed=seed, inner_val_fraction=frac)
    del _train_pos  # inner-val only — never materialise the train slice
    if len(inner_val_pos) > cap:
        rng = np.random.RandomState(int(seed) ^ INNER_VAL_CAP_SALT)
        inner_val_pos = sorted(rng.choice(
            list(inner_val_pos), size=cap, replace=False).tolist())
    inner_val_pos = [int(p) for p in inner_val_pos]

    # denominators MUST match the historical run (fail-closed)
    curves = sorted(_curve_id_set_of(corpus["dataset"].samples, inner_val_pos))
    hist_curves = sorted(str(c) for c in (history.get("inner_val_curves") or []))
    gates = [
        _compare("inner_val_n_positions", len(inner_val_pos),
                 int(history.get("n_inner_val_samples") or -1), 0, "exact"),
        _compare("inner_val_curve_set_sha",
                 _sha256_text(json.dumps(curves, separators=(",", ":"))),
                 _sha256_text(json.dumps(hist_curves, separators=(",", ":"))),
                 0, "exact"),
    ]
    denom = {
        "n_positions": len(inner_val_pos),
        "n_curves": len(curves),
        "inner_val_fraction": frac,
        "inner_val_cap": cap,
        "positions_sha256": _canonical_positions_sha(inner_val_pos),
        "curves_sha256": _canonical_curves_sha(curves),
        "history_match": _gate_flags(gates),
        "gates": gates,
    }
    if not _pass_all(gates):
        raise RepairError(
            "INNER_VAL_DENOMINATOR_MISMATCH",
            "seed %d rebuild does not match history: %s" % (
                seed, json.dumps(_gate_flags(gates), sort_keys=True)))
    return inner_val_pos, denom


# ---------------------------------------------------------------------------
# fidelity / denominator gate builders
# ---------------------------------------------------------------------------


def _dev_fidelity_gates(move: Mapping[str, Any],
                        process: Mapping[str, Any],
                        history: Mapping[str, Any],
                        probe: Mapping[str, Any],
                        tie_tols: Optional[Mapping[str, Any]],
                        ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Returns (gated, recorded) fidelity checks vs historical metrics.json.

    Pair-independent metrics (q / endpoint) are gated at the frozen
    paired-CI tolerances — same forward composition as history.
    Ordering accuracy has NO frozen tolerance (the paired-CI gate never
    contained it): it is gated at the DERIVED float-noise bound from
    ``_ordering_tie_tols`` (inner-val probe noise + this run's q ULPs —
    independent of the historical diff); its pair COUNTS stay exact.
    RMSE/zero + cos use the documented cross-composition band.  The legacy
    pair SET (n_sign/n_active, u-independent) is gated EXACTLY; legacy pair
    OUTCOMES (dead count at the probe-derived tolerance, sign/SRE) are
    gated only when the forward-drift probe proves composition invariance,
    otherwise recorded (values stay in the sealed files only).
    """
    hm = history["history_move_metrics"]
    h_rmse = hm.get("key_rmse_over_zero") or {}
    h_sign = hm.get("sre_and_sign") or {}
    h_q = hm.get("q_head") or {}
    h_ord = hm.get("ordering") or {}
    h_ep = hm.get("endpoint_head") or {}
    after = move["after_repaired"]
    before = move["before_legacy"]
    a_rmse = after["key_rmse_over_zero"]
    b_sgn = before["sre_and_sign"]
    q = process["q_head"]
    ordn = process["ordering"]
    ep = process["endpoint"]
    invariant = bool(probe.get("u_composition_invariant"))
    dead_tol = int(probe.get("dead_hist_tol", 2))
    hist_active = int(h_sign.get("n_active_pairs") or 0)

    # derived ordering tolerances (float-noise bound); fail-closed if the
    # probe/tie-stats are missing — ordering must not fall back to a
    # guessed flat tolerance.
    if not tie_tols or tie_tols.get("tol_ordering_full") is None:
        raise RepairError(
            "ORDERING_TOLERANCE_MISSING",
            "derived ordering tie tolerances unavailable (probe missing?)")
    tol_full = float(tie_tols["tol_ordering_full"])
    tol_cross = float(tie_tols.get("tol_ordering_cross_ts") or tol_full)
    tie_note = ("derived float-noise bound: flippable/n_pairs with band "
                "4*q_noise+2*(ULP32 q_i+q_j); inputs independent of the "
                "history diff")

    gated: List[Dict[str, Any]] = [
        # pair-independent, SAME forward composition as history:
        _compare("q_mae_curve_mean", q.get("mae_curve_mean"),
                 h_q.get("mae_curve_mean"), TOL["q_mae_curve_mean"], "abs"),
        _compare("q_corr_curve_mean", q.get("corr_curve_mean"),
                 h_q.get("corr_curve_mean"), TOL["q_corr_curve_mean"], "abs"),
        _compare("q_n_eval_curves", q.get("n_eval_curves"),
                 h_q.get("n_eval_curves"), 0, "exact"),
        _compare("ordering_accuracy_full",
                 ordn.get("pairwise_accuracy_full"),
                 h_ord.get("pairwise_accuracy_full"),
                 tol_full, "abs"),
        _compare("ordering_n_pairs_full", ordn.get("n_pairs_full"),
                 h_ord.get("n_pairs_full"), 0, "exact"),
        _compare("ordering_accuracy_cross_ts",
                 ordn.get("pairwise_accuracy_cross_ts"),
                 h_ord.get("pairwise_accuracy_cross_ts"),
                 tol_cross, "abs"),
        _compare("ordering_n_pairs_cross_ts", ordn.get("n_pairs_cross_ts"),
                 h_ord.get("n_pairs_cross_ts"), 0, "exact"),
        # counted denominators must match the derived tie stats exactly
        _compare("ordering_tie_n_pairs_full",
                 tie_tols.get("n_pairs_full"), ordn.get("n_pairs_full"),
                 0, "exact"),
        _compare("ordering_tie_n_pairs_cross_ts",
                 tie_tols.get("n_pairs_cross_ts"), ordn.get("n_pairs_cross_ts"),
                 0, "exact"),
        _compare("endpoint_auprc", ep.get("auprc"), h_ep.get("auprc"),
                 TOL["endpoint_auprc"], "abs"),
        _compare("endpoint_auroc", ep.get("auroc"), h_ep.get("auroc"),
                 TOL["endpoint_auroc"], "abs"),
        _compare("endpoint_prevalence", ep.get("prevalence"),
                 h_ep.get("prevalence"), TOL["endpoint_prevalence"], "abs"),
        _compare("endpoint_n_unmasked", ep.get("n_unmasked"),
                 h_ep.get("n_unmasked"), 0, "exact"),
        # pair-independent, cross-forward-composition band:
        _compare("median_rmse_over_zero", a_rmse.get("median_rmse_over_zero"),
                 h_rmse.get("median_rmse_over_zero"),
                 TOL["median_rmse_over_zero"], "abs"),
        _compare("cos_eff_median", a_rmse.get("cos_eff_median"),
                 h_rmse.get("cos_eff_median"), TOL["cos_eff_median"], "abs"),
        # legacy pair SET (u-independent → EXACT):
        _compare("legacy_n_sign_pairs", b_sgn.get("n_sign_pairs"),
                 h_sign.get("n_sign_pairs"), 0, "exact"),
        _compare("legacy_n_active_pairs", b_sgn.get("n_active_pairs"),
                 h_sign.get("n_active_pairs"), 0, "exact"),
        _compare("legacy_n_clusters_sign", b_sgn.get("n_clusters_sign"),
                 h_sign.get("n_clusters_sign"), 0, "exact"),
        _compare("legacy_n_sre_frames", b_sgn.get("n_sre_frames"),
                 h_sign.get("n_sre_frames"), 0, "exact"),
        # legacy pair OUTCOMES: dead count at the probe-derived tolerance
        _compare("legacy_n_dead_pairs", b_sgn.get("n_dead_pairs"),
                 h_sign.get("n_dead_pairs"), dead_tol, "abs"),
        _compare("legacy_dead_fraction", b_sgn.get("dead_zone_fraction"),
                 h_sign.get("dead_zone_fraction"),
                 (float(dead_tol) / hist_active) if hist_active else 1.0,
                 "abs"),
    ]
    for g in gated:
        if g["name"].startswith("ordering_accuracy"):
            g["tol_source"] = tie_note
    recorded: List[Dict[str, Any]] = []
    if invariant:
        gated.append(_compare(
            "legacy_sign_median", b_sgn.get("sign_accuracy_cluster_median"),
            h_sign.get("sign_accuracy_cluster_median"),
            TOL["legacy_sign_median"], "abs"))
        gated.append(_compare(
            "legacy_sre_median", b_sgn.get("sre_cluster_median"),
            h_sign.get("sre_cluster_median"), TOL["legacy_sre_median"], "abs"))
    else:
        # u drift measured → outcome values recorded (sealed) not gated:
        recorded.append(_record(
            "legacy_sign_median", b_sgn.get("sign_accuracy_cluster_median"),
            h_sign.get("sign_accuracy_cluster_median")))
        recorded.append(_record(
            "legacy_sre_median", b_sgn.get("sre_cluster_median"),
            h_sign.get("sre_cluster_median")))
    return gated, recorded


def _dev_denominator_gates(history: Mapping[str, Any], n_dev: int,
                           n_ok: int, n_dev_curves: int, n_move_curves: int,
                           positions_sha: str,
                           prov_pack: Mapping[str, Any],
                           ) -> List[Dict[str, Any]]:
    return [
        _compare("n_dev_positions", n_dev,
                 int(history.get("n_dev_samples") or -1), 0, "exact"),
        _compare("n_move_ok_positions", n_ok,
                 int(history.get("move_n_val_samples") or -1), 0, "exact"),
        _compare("n_dev_curves", n_dev_curves,
                 int(history.get("n_dev_curves") or -1), 0, "exact"),
        _compare("n_move_curves", n_move_curves,
                 int(history.get("move_n_val_curves") or -1), 0, "exact"),
        _compare("dev_positions_sha_vs_provenance", positions_sha,
                 (prov_pack or {}).get("positions_sha256"), 0, "exact"),
    ]


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------


def _load_c_model(seed: int, device: str) -> Tuple[Any, Dict[str, Any]]:
    import torch

    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        load_model_v53mp,
    )
    from scripts.campaigns.r3_v53mp_run import R3_WINNER_CKPT

    path = _c_ckpt(seed)
    state = torch.load(str(path), map_location=device, weights_only=False)
    cfg = _trainer_config(state, device)
    model = load_model_v53mp(cfg, r3_ckpt=R3_WINNER_CKPT, device=device)
    model.load_state_dict(state["model"], strict=True)
    return model, state


def _load_r3_model(device: str) -> Any:
    import torch

    from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3

    payload = torch.load(str(R3_CKPT), map_location="cpu", weights_only=False)
    if not (isinstance(payload, Mapping) and "model" in payload):
        raise RepairError("R3_CHECKPOINT_FORMAT", str(R3_CKPT))
    model = StarEGNNV3()
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device)


def _iv_cache(dataset: Any) -> Any:
    """Pinned inner-val feature cache (built once per process, shared by
    every model of the stage — features are model-independent)."""
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        FEATURE_LEVEL_V53MP,
        _CacheView,
        _FeatureLRU,
    )
    return _CacheView(_FeatureLRU(
        dataset, capacity=INNER_VAL_CAP_DEFAULT + 64,
        level=FEATURE_LEVEL_V53MP, intent_off=False))


def _eval_split(
    split: str,
    positions: Sequence[int],
    *,
    corpus: Mapping[str, Any],
    cache: Any,
    device: str,
    model: Any,
    collect_process: bool,
    tau_end: Optional[float],
    cost: _Cost,
    ordering_q_noise: Optional[float] = None,
) -> Dict[str, Any]:
    """ONE prediction pass + both g53 variants + process metrics.

    ``ordering_q_noise`` (inner-val probe measurement) enables the DERIVED
    ordering tolerances for this split; None (inner-val stage) skips them.
    """
    preds, stats = _predict_once(
        model, positions, cache, device,
        collect_process=collect_process,
        labels=corpus.get("labels"), endpoint=corpus.get("endpoint"))
    cost.note_forward(stats["forward_wall_s"], stats["n_forward_batches"],
                      stats["n_positions"])
    ok_mask = _ok_mask(corpus["dataset"], positions)
    move_positions = [int(p) for p, m in zip(positions, ok_mask) if m]
    if not move_positions:
        raise RepairError("MOVE_EMPTY", "%s split has no ok positions" % split)
    legacy_mv, repaired_mv, mv_meta = _build_move_variants(preds, move_positions)
    move = _move_metrics(legacy_mv, repaired_mv, mv_meta)
    process: Optional[Dict[str, Any]] = None
    tie_tols: Optional[Dict[str, Any]] = None
    if collect_process:
        ordered = [preds[int(p)] for p in positions]
        process = _process_metrics(ordered, tau_end=tau_end)
        if ordering_q_noise is not None:
            tie_tols = _ordering_tie_tols(ordered, float(ordering_q_noise))
    n_curves_all = len({str(p.curve_id) for p in preds.values()})
    n_curves_move = len({preds[int(p)].curve_id for p in move_positions})
    return {
        "denominators": {
            "n_positions": len(positions),
            "n_positions_ok": len(move_positions),
            "n_curves": n_curves_all,
            "n_curves_move": n_curves_move,
            "positions_sha256": _canonical_positions_sha(positions),
            "curves_sha256": _canonical_curves_sha(
                sorted({str(p.curve_id) for p in preds.values()})),
            "n_forward_batches": stats["n_forward_batches"],
        },
        "move": move,
        "process": process,
        "ordering_tie_tols": tie_tols,
        "sanity_gates": _move_sanity(
            move, n_positions=len(positions),
            n_positions_ok=len(move_positions),
            n_curves_move=n_curves_move),
    }


# ---------------------------------------------------------------------------
# forward-drift probe (smoke) — derives the dev dead-count tolerance
# ---------------------------------------------------------------------------


def _forward_drift_probe(model: Any, probe_positions: Sequence[int],
                         cache: Any, device: str, *,
                         labels: Any, endpoint: Any,
                         hist_move_n_ok: int) -> Dict[str, Any]:
    """Two forward compositions on the same inner-val positions.

    Measures the float32 drift caused purely by batch composition (the
    single required dev pass uses all-positions chunking; the historical
    MOVE pass used ok-only chunking):

    * ``max_abs_u_diff`` → u-dependent MOVE dead-count tolerance
      (boundary pairs within the sensitivity band, scaled to dev size);
    * ``max_abs_q_diff`` → the noise term of the DERIVED ordering-accuracy
      tolerance (``_ordering_tie_tols``) — measured on q directly, never
      inferred from the historical comparison.
    """
    from pes2ts_core.pesn.offline.evaluation.key_atom_protocols import (
        G53_SIGN_PAIR_THRESHOLD,
        exact_delta_d,
    )

    a, _ = _predict_once(model, probe_positions, cache, device,
                         collect_process=True, labels=labels,
                         endpoint=endpoint, batch_size=MOVE_BATCH)
    b, _ = _predict_once(model, probe_positions, cache, device,
                         collect_process=True, labels=labels,
                         endpoint=endpoint,
                         batch_size=max(len(probe_positions), MOVE_BATCH + 1))
    thr = float(G53_SIGN_PAIR_THRESHOLD)
    noise = 0.0
    q_noise = 0.0
    ep_noise = 0.0
    n_pairs = 0
    dead_flips = 0
    correctness_flips = 0
    boundary = 0
    for p in probe_positions:
        ra, rb = a[int(p)], b[int(p)]
        if ra.u.size:
            noise = max(noise, float(np.max(np.abs(
                ra.u.astype(np.float64) - rb.u.astype(np.float64)))))
        q_noise = max(q_noise, abs(float(ra.q_pred) - float(rb.q_pred)))
        ep_noise = max(ep_noise, abs(float(ra.ep_logit) - float(rb.ep_logit)))
        slot = {int(r): k for k, r in enumerate(ra.key_rows)}
        cols = [(int(ra.pairs[0, c]), int(ra.pairs[1, c]))
                for c in range(ra.pairs.shape[1])]
        kp = [(i, j) for (i, j) in cols if i in slot and j in slot]
        if not kp:
            continue
        pos = ra.pos.astype(np.float64)
        krows = [int(r) for r in ra.key_rows]
        dd_a = exact_delta_d(pos, krows, kp, ra.u.astype(np.float64))
        dd_b = exact_delta_d(pos, krows, kp, rb.u.astype(np.float64))
        dd_y = exact_delta_d(pos, krows, kp, ra.y.astype(np.float64))
        n_pairs += len(kp)
        dead_a = np.abs(dd_a) < thr
        dead_b = np.abs(dd_b) < thr
        dead_flips += int(np.sum(dead_a != dead_b))
        sig = np.abs(dd_y) >= thr
        corr_a = (np.sign(dd_a) == np.sign(dd_y)) & (np.abs(dd_a) >= thr)
        corr_b = (np.sign(dd_b) == np.sign(dd_y)) & (np.abs(dd_b) >= thr)
        correctness_flips += int(np.sum((corr_a != corr_b) & sig))
        band = max(4.0 * noise, 1e-12)
        boundary += int(np.sum(np.abs(np.abs(dd_a) - thr) <= band))
    scale = int(np.ceil(max(hist_move_n_ok, 1) / max(len(probe_positions), 1)))
    dead_hist_tol = max(2, int(np.ceil(boundary * scale)) + dead_flips)
    invariant = noise == 0.0
    return {
        "schema": "v541_forward_drift_probe_v1",
        "n_probe_positions": len(probe_positions),
        "n_probe_pairs": n_pairs,
        "max_abs_u_diff": noise,
        "max_abs_q_diff": q_noise,
        "max_abs_ep_logit_diff": ep_noise,
        "u_composition_invariant": invariant,
        "dead_classification_flips": dead_flips,
        "correctness_flips": correctness_flips,
        "boundary_pairs_at_sign_threshold": boundary,
        "dev_scale_factor": scale,
        "dead_hist_tol": dead_hist_tol,
        "note": ("two forward compositions (chunk64 vs one batch) on "
                 "inner-val; dead_hist_tol = boundary pairs scaled to the "
                 "dev ok-position count, floor 2; max_abs_q_diff feeds the "
                 "derived ordering tolerance (_ordering_tie_tols)"),
    }


def _ordering_tie_tols(preds: Sequence["_PosPred"],
                       q_noise: float) -> Dict[str, Any]:
    """DERIVED ordering-accuracy tolerances (never from the history diff).

    Pairwise ordering accuracy compares float32 q values: a comparison
    ``qp[later] > qp[earlier]`` can flip between two runs only when the
    two values sit within their combined uncertainty

        band(i, j) = 4 * q_noise + 2 * (ULP32(q_i) + ULP32(q_j))

    where ``q_noise`` is the inner-val measured cross-composition q drift
    (probe) and ULP32 covers float32 representation.  The tolerance is

        tol = (# pairs with |qp_i - qp_j| <= band) / n_pairs

    i.e. a PROVABLE upper bound on the accuracy change float noise can
    cause, derived only from INDEPENDENT inputs (inner-val probe noise +
    this run's own q values) — never from the historical comparison.
    Companion gates (n_pairs/n_eval exact, q_mae/q_corr frozen) still pin
    the prediction path structurally.

    Mirrors ``ordering_pair_accuracy``'s pair enumeration exactly (gap=0,
    cross_ts filter) so the counted denominators match the gated ones.
    """
    import math

    noise = max(float(q_noise), 0.0)
    groups = [(str(p.curve_id), str(p.direction)) for p in preds]
    by_group: Dict[Tuple[str, str], List[int]] = {}
    for i, p in enumerate(preds):
        if p.q_mask:
            by_group.setdefault(groups[i], []).append(i)

    def _ulp32(v: float) -> float:
        if v == 0.0:
            return math.exp2(-24)  # conservative floor for exact zeros
        _mant, exp = math.frexp(abs(v))   # v = mant * 2**exp, mant in [0.5,1)
        return math.exp2(exp - 24)        # float32 ULP = 2**(binade-23)

    n_full = flippable_full = 0
    n_cross = flippable_cross = 0
    for idxs in by_group.values():
        qp = [float(preds[i].q_pred) for i in idxs]
        qt = [float(preds[i].q_true) for i in idxs]
        ulps = [_ulp32(v) for v in qp]
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                qi, qj = qt[a], qt[b]
                if qj > qi:
                    earlier, later = a, b
                elif qi > qj:
                    earlier, later = b, a
                else:
                    continue  # equal truths: not a pair in the metric
                gap = abs(qp[a] - qp[b])
                band = 4.0 * noise + 2.0 * (ulps[a] + ulps[b])
                flippable = gap <= band
                n_full += 1
                flippable_full += 1 if flippable else 0
                if qt[earlier] * qt[later] < 0.0:   # cross_ts subset
                    n_cross += 1
                    flippable_cross += 1 if flippable else 0
    tol_full = (max(flippable_full / n_full, 1e-9) if n_full else None)
    tol_cross = (max(flippable_cross / n_cross, 1e-9) if n_cross else None)
    return {
        "tol_ordering_full": tol_full,
        "tol_ordering_cross_ts": tol_cross,
        "n_pairs_full": n_full,
        "n_flippable_full": flippable_full,
        "n_pairs_cross_ts": n_cross,
        "n_flippable_cross_ts": flippable_cross,
        "q_noise": noise,
        "band_formula": ("4*q_noise + 2*(ULP32(q_i)+ULP32(q_j)); "
                         "tol = flippable/n_pairs (provable float-noise "
                         "bound; inputs independent of the history diff)"),
    }


# ---------------------------------------------------------------------------
# stage: smoke (fixtures + ONE inner-val position vs BOTH prod entries)
# ---------------------------------------------------------------------------


def _fixture_samples(n_nodes: Sequence[int], n_keys: Sequence[int],
                     key_index: Sequence[int],
                     pairs: Sequence[Tuple[int, int]],
                     edits: Sequence[float],
                     curve_ids: Sequence[str]):
    """Hand-made GROUP arrays (global rows) for the two g53 builders."""
    key_index_a = np.asarray(key_index, dtype=np.int64)
    pair_index = (np.asarray(pairs, dtype=np.int64).T.reshape(2, -1)
                  if len(pairs) else np.zeros((2, 0), dtype=np.int64))
    pair_edit = np.asarray(edits, dtype=np.float32)
    n_nodes_a = np.asarray(n_nodes, dtype=np.int64)
    n_keys_a = np.asarray(n_keys, dtype=np.int64)
    n_total = int(n_nodes_a.sum())
    n_k = int(n_keys_a.sum())
    pos_all = (np.arange(n_total, dtype=np.float32).reshape(-1, 1)
               * np.array([1.0, 0.5, 0.25], dtype=np.float32))
    # y with a per-key gradient → every pair has |Δd(y)| >> 0.005 A so the
    # sign denominator reflects the PAIR SET (the thing under audit).
    y_all = np.zeros((n_k, 3), dtype=np.float64)
    y_all[:, 0] = 0.1 * (np.arange(n_k, dtype=np.float64) + 1.0)
    u_all = np.zeros((n_k, 3), dtype=np.float64)
    ownership = assign_batch_pairs(
        n_nodes=n_nodes_a, n_keys=n_keys_a, key_index=key_index_a,
        pair_index=pair_index, pair_edit=pair_edit)
    legacy = _legacy_g53_samples(
        n_keys_a, key_index_a, pair_index, pair_edit,
        pos_all, u_all, y_all, curve_ids)
    repaired = _repaired_g53_samples(
        ownership, pos_all, u_all, y_all, curve_ids, key_index_a)
    return legacy, repaired, ownership


def cmd_smoke(_args: argparse.Namespace) -> int:
    """Fixture proof + ONE real inner-val position vs BOTH prod entries
    + forward-drift probe (no dev access anywhere in this stage)."""
    from pes2ts_core.pesn.offline.evaluation.key_atom_protocols import (
        evaluate_g53,
    )

    report: Dict[str, Any] = {"schema": SCHEMA_SMOKE, "created_at": _now(),
                              "checks": {}, "new_QC_calls": 0}
    checks: Dict[str, Any] = report["checks"]
    ok = True

    def _check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        checks[name] = {"pass": bool(cond), "detail": detail}
        ok = ok and bool(cond)
        _say("SMOKE %s: %s%s" % (name, "PASS" if cond else "FAIL",
                                 (" — " + detail) if detail else ""))

    # --- fixture 1: the WP1a old-fail sample (legacy drops graph1) -------
    # group: pos0 nodes 0..5 (6), keys {0,1}; pos1 nodes 6..8 (3), keys {6,8}
    legacy, repaired, _own = _fixture_samples(
        n_nodes=[6, 3], n_keys=[2, 2], key_index=[0, 1, 6, 8],
        pairs=[(0, 1), (6, 8)], edits=[0.4, -0.4],
        curve_ids=["FIX_A", "FIX_B"])
    legacy_pairs = [s["pairs"] for s in legacy]
    repaired_pairs = [s["pairs"] for s in repaired]
    _check("fixture_legacy_drops_graph1_pair",
           legacy_pairs == [[(0, 1)], []],
           "legacy kept %s" % (legacy_pairs,))
    _check("fixture_repaired_keeps_both_pairs",
           repaired_pairs == [[(0, 1)], [(6, 8)]],
           "repaired kept %s" % (repaired_pairs,))
    n_sign_leg = evaluate_g53(legacy)["aggregate"]["n_sign_pairs"]
    n_sign_rep = evaluate_g53(repaired)["aggregate"]["n_sign_pairs"]
    _check("fixture_sign_denominator_recovers_pair",
           n_sign_rep == n_sign_leg + 1,
           "sign denominators legacy=%d repaired=%d (expect +1)"
           % (n_sign_leg, n_sign_rep))

    # --- fixture 2: legacy AGREES with repaired on the prefix case -------
    legacy2, repaired2, _ = _fixture_samples(
        n_nodes=[2, 2], n_keys=[2, 2], key_index=[0, 1, 2, 3],
        pairs=[(0, 1), (2, 3)], edits=[0.5, 0.5],
        curve_ids=["FIX_C", "FIX_D"])
    _check("fixture_legacy_agrees_on_prefix_case",
           [s["pairs"] for s in legacy2] == [s["pairs"] for s in repaired2],
           "both variants collect %s" % ([s["pairs"] for s in repaired2],))
    _check("fixture_g53_structure_unchanged",
           set(repaired[0].keys())
           == {"pos", "key_rows", "pairs", "pair_edits", "u", "y",
               "group_id"},
           "g53 sample contract keys preserved")

    # --- ONE real inner-val position vs BOTH production entries -----------
    try:
        cost = _Cost()
        lock = _GpuLock()
        lock.acquire()
        try:
            corpus = _open_corpus_light()
            m17 = _read_json(_c_metrics(17))
            hist = {
                "n_inner_val_samples": m17.get("n_inner_val_samples"),
                "inner_val_curves": m17.get("inner_val_curves"),
                "inner_val_max_samples": m17.get("inner_val_max_samples"),
            }
            device = "cuda"
            model, state = _load_c_model(17, device)
            try:
                iv_pos, iv_denom = _rebuild_inner_val(
                    corpus, 17, state.get("config", {}), hist)
                ok_mask = _ok_mask(corpus["dataset"], iv_pos)
                ok_positions = [int(p) for p, m in zip(iv_pos, ok_mask) if m]
                if not ok_positions:
                    raise RepairError("SMOKE_NO_OK_POSITION",
                                      "inner-val has no ok positions")
                p0 = ok_positions[0]
                cache = _iv_cache(corpus["dataset"])

                # audit pipeline on the SAME single position (one batch)
                preds, stats = _predict_once(
                    model, [p0], cache, device,
                    collect_process=True, labels=corpus["labels"],
                    endpoint=corpus["endpoint"])
                legacy_mv, repaired_mv, mv_meta = _build_move_variants(
                    preds, [p0])
                move = _move_metrics(legacy_mv, repaired_mv, mv_meta)

                # production entry 1: evaluate_checkpoint (WP1a shared tool)
                from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import (
                    evaluate_checkpoint,
                )
                prod1 = evaluate_checkpoint(
                    model, corpus["dataset"], cache, [p0],
                    device=device, bootstrap_draws=0)

                # production entry 2: the paired-CI _collect
                col = _collect(model, cache, [p0], corpus["labels"],
                               corpus["endpoint"], device)

                def _close(x: Any, y: Any, tol: float = 1e-6) -> bool:
                    """Counts/structures exact; floats within the measured
                    GPU pass-to-pass band (u noise ~1e-8 → metric noise
                    ≲2e-7; 1e-6 matches the frozen paired-CI tolerances)."""
                    if isinstance(x, dict) and isinstance(y, dict):
                        return (set(x) == set(y)
                                and all(_close(x[k], y[k], tol) for k in x))
                    if isinstance(x, (list, tuple)) and isinstance(
                            y, (list, tuple)):
                        return len(x) == len(y) and all(
                            _close(p, q, tol) for p, q in zip(x, y))
                    if isinstance(x, bool) or isinstance(y, bool):
                        return x == y
                    if isinstance(x, (int, np.integer)) and isinstance(
                            y, (int, np.integer)):
                        return int(x) == int(y)
                    if x is None or y is None:
                        return x is y
                    if isinstance(x, (float, np.floating)) or isinstance(
                            y, (float, np.floating)):
                        return abs(float(x) - float(y)) <= tol
                    return x == y

                _check("audit_repaired_matches_evaluate_checkpoint_sre",
                       _close(move["after_repaired"]["sre_and_sign"],
                              prod1["sre_and_sign"]),
                       "entry point 1 vs audit-after (sre_and_sign)")
                _check("audit_repaired_matches_evaluate_checkpoint_rmse",
                       _close(move["after_repaired"]["key_rmse_over_zero"],
                              prod1["key_rmse_over_zero"]),
                       "entry point 1 vs audit-after (g52)")
                _check("audit_pair_audit_matches_evaluate_checkpoint",
                       _close(move["pair_denominators"]["pair_ownership_audit"],
                              prod1.get("pair_ownership_audit")),
                       "pair_ownership_audit totals identical")
                _check("audit_repaired_pairs_match_collect_entry",
                       [s["pairs"] for s in repaired_mv]
                       == [s["pairs"] for s in col["g53_samples"]]
                       and [s["pair_edits"] for s in repaired_mv]
                       == [s["pair_edits"] for s in col["g53_samples"]]
                       and [s["key_rows"] for s in repaired_mv]
                       == [s["key_rows"] for s in col["g53_samples"]],
                       "entry point 2 pairs/edits/key_rows identical")
                _check("audit_pair_audit_matches_collect",
                       _close(move["pair_denominators"]["pair_ownership_audit"],
                              col.get("pair_ownership_audit")),
                       "pair_ownership_audit totals identical")
                # batch-of-1: legacy ⊆ repaired (equality needs prefix keys)
                subset_ok = all(
                    set(lg["pairs"]).issubset(set(rp["pairs"]))
                    for lg, rp in zip(legacy_mv, repaired_mv))
                _check("single_position_legacy_subset_repaired", subset_ok,
                       "legacy ⊆ repaired on batch-of-1 (%d pairs)"
                       % len(repaired_mv[0]["pairs"]))
                _check("smoke_denominators_recorded",
                       iv_denom["n_positions"] == 8192
                       and len(ok_positions) > 0,
                       "inner-val positions=%d ok=%d p0=%d" % (
                           iv_denom["n_positions"], len(ok_positions), p0))

                # forward-drift probe → dev gate tolerances (u dead-count
                # band + q-noise term of the derived ordering tolerance)
                probe_list = ok_positions[:PROBE_POSITIONS]
                probe = _forward_drift_probe(
                    model, probe_list, cache, device,
                    labels=corpus["labels"], endpoint=corpus["endpoint"],
                    hist_move_n_ok=int(
                        (m17.get("move") or {}).get("n_val_samples") or 1))
                _check("probe_noise_sane",
                       float(probe["max_abs_u_diff"]) <= PROBE_NOISE_SANITY_MAX
                       and float(probe["max_abs_q_diff"]) <= PROBE_NOISE_SANITY_MAX,
                       "max_abs_u_diff=%.3e max_abs_q_diff=%.3e (cap %.0e)" % (
                           probe["max_abs_u_diff"], probe["max_abs_q_diff"],
                           PROBE_NOISE_SANITY_MAX))
                report["forward_drift_probe"] = probe
                report["inner_val_denominators"] = {
                    "n_positions": iv_denom["n_positions"],
                    "n_ok_positions": len(ok_positions),
                    "n_curves": iv_denom["n_curves"],
                    "positions_sha256": iv_denom["positions_sha256"],
                }
                cost.note_forward(stats["forward_wall_s"],
                                  stats["n_forward_batches"], 1)
                report["cost"] = cost.snapshot(device)
            finally:
                del model
                _free_cuda()
        finally:
            lock.release()
    except Exception as exc:  # fail-closed: record, don't crash untraced
        _check("real_position_smoke", False,
               "%s: %s" % (type(exc).__name__, exc))
        report["failure"] = {"error": "%s: %s" % (type(exc).__name__, exc),
                             "traceback": traceback.format_exc()[-4000:]}
        ok = False

    report["pass"] = bool(ok)
    sha = _write_json(SMOKE_REPORT, report)
    _say("smoke %s → %s (sha256=%s)" % (
        "PASS" if ok else "FAIL", SMOKE_REPORT, sha[:16]))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# stage: run (inner-val readable | dev sealed)
# ---------------------------------------------------------------------------


def _run_inner_val(seeds: Sequence[int], device: str,
                   cost: _Cost) -> None:
    if not SMOKE_REPORT.is_file():
        raise RepairError("SMOKE_FIRST", "run the smoke stage first")
    smoke = _read_json(SMOKE_REPORT)
    if not smoke.get("pass"):
        raise RepairError("SMOKE_FAILED", "smoke_report pass=false")
    freeze = _freeze_assets()
    corpus = _open_corpus_light()
    cache = _iv_cache(corpus["dataset"])
    r3_model = _load_r3_model(device)   # ONE load, reused for every seed
    try:
        doc = (_read_json(INNER_VAL_JSON)
               if INNER_VAL_JSON.is_file() else None)
        out: Dict[str, Any] = (
            doc if doc and doc.get("schema") == SCHEMA_INNER_VAL else {
                "schema": SCHEMA_INNER_VAL,
                "created_at": _now(),
                "wp": "WP1b",
                "purpose": "baseline_repair",
                "note": ("inner-val results are READABLE; dev details are "
                         "sealed in dev_baseline_sealed/ — no unseal before "
                         "WP7 (plan §4.3/§9.3)"),
            })
        out["assets_frozen"] = freeze
        out["smoke_report_sha256"] = _sha256_file(SMOKE_REPORT)
        out["metric_definitions"] = {
            "before_legacy": ("pre-WP1a key-ptr g53 replica (audit-only; "
                              "the algorithm removed from production by "
                              "WP1a — never re-imported)"),
            "after_repaired": (
                "WP1a shared node-ptr g53 (pair_ownership_v3."
                "assign_batch_pairs — the same object both production "
                "entry points import); FROZEN as the sole future baseline"),
            "rmse_zero": "G5.2 median_rmse_over_zero (pair-independent)",
            "sign": "G5.3 sign_accuracy_cluster_median (teacher-significant "
                    "pairs; n_sign_pairs/n_active_pairs are u-independent)",
            "sre": "G5.3 sre_cluster_median",
            "dead": "G5.3 dead_zone_fraction (|Δd(u)|<0.005 A abstention)",
            "fixed_coverage_direction": (
                "G5.2 effective-atom soft-cosine median at TEACHER-fixed "
                "coverage (||y||>=0.02 A); pair-independent"),
            "pair_denominators": ("n_pairs_input / collected_before / "
                                  "collected_after + WP1a audit totals"),
            "prediction_passes_per_checkpoint_per_split": 1,
            "q_ordering_endpoint": (
                "computed once from the same prediction; pair-independent "
                "(before == after by construction); fidelity-gated on dev"),
        }
        out["inner_val"] = out.get("inner_val", {})
        for seed in seeds:
            _say("=== inner-val seed %d ===" % seed)
            hist_block = freeze["c_runs"][str(seed)]
            hist = hist_block["history_denominators"]
            model, state = _load_c_model(seed, device)
            try:
                iv_pos, iv_denom = _rebuild_inner_val(
                    corpus, seed, state.get("config", {}), hist)
                _say("inner-val seed %d: positions=%d curves=%d sha=%s" % (
                    seed, iv_denom["n_positions"], iv_denom["n_curves"],
                    iv_denom["positions_sha256"][:16]))
                tau = ((state.get("tau_end_calibration") or {}).get("tau_end"))
                c_res = _eval_split(
                    "inner-val", iv_pos, corpus=corpus, cache=cache,
                    device=device, model=model, collect_process=True,
                    tau_end=tau, cost=cost)
            finally:
                del model
                _free_cuda()

            # R3 reference on the SAME seed inner-val (MOVE only, H2)
            r3_res = _eval_split(
                "inner-val", iv_pos, corpus=corpus, cache=cache,
                device=device, model=r3_model, collect_process=False,
                tau_end=None, cost=cost)

            h2 = [
                _h2_move_guard(c_res["move"]["after_repaired"],
                               r3_res["move"]["after_repaired"],
                               split="inner-val", variant="after_repaired",
                               seed=seed),
                _h2_move_guard(c_res["move"]["before_legacy"],
                               r3_res["move"]["before_legacy"],
                               split="inner-val", variant="before_legacy",
                               seed=seed),
            ]
            seed_ok = (_pass_all(c_res["sanity_gates"])
                       and _pass_all(r3_res["sanity_gates"])
                       and _pass_all(iv_denom["gates"]))
            out["inner_val"][str(seed)] = {
                "denominators": iv_denom,
                "C_s%d" % seed: {
                    "checkpoint_sha256":
                        hist_block["checkpoint"]["sha256"],
                    "denominators": c_res["denominators"],
                    "move": c_res["move"],
                    "process": c_res["process"],
                    "process_pair_fix_unchanged": True,  # never sees pairs
                    "sanity_gates": c_res["sanity_gates"],
                    "sanity_pass": _pass_all(c_res["sanity_gates"]),
                },
                "r3_reference": {
                    "checkpoint_sha256":
                        freeze["r3_reference"]["checkpoint"]["sha256"],
                    "denominators": r3_res["denominators"],
                    "move": r3_res["move"],
                    "sanity_gates": r3_res["sanity_gates"],
                    "sanity_pass": _pass_all(r3_res["sanity_gates"]),
                    "process": None,
                    "note": "H2 MOVE reference only (R3 has no heads)",
                },
                "h2_move_guard_inner_val": h2,
                "split_pass": seed_ok,
            }
            _say("inner-val seed %d: sanity %s | pairs %d→%d | sign_pairs %d→%d" % (
                seed, "PASS" if seed_ok else "FAIL",
                c_res["move"]["pair_denominators"]["n_pairs_collected_before"],
                c_res["move"]["pair_denominators"]["n_pairs_collected_after"],
                int(c_res["move"]["before_legacy"]["sre_and_sign"]
                    .get("n_sign_pairs") or 0),
                int(c_res["move"]["after_repaired"]["sre_and_sign"]
                    .get("n_sign_pairs") or 0)))
            del state
            _free_cuda()
        del r3_model
        _free_cuda()

        # overall inner-val status over ALL seeds present in the document
        all_ok = all(bool(b.get("split_pass"))
                     for b in out["inner_val"].values())
        out["status"] = "ok" if all_ok else "gates_failed"
        out.setdefault("dev_sealed_status", {"status": "pending"})
        out["cost"] = out.get("cost", {})
        out["cost"]["inner_val_stage"] = cost.snapshot(device)
        out["cost"]["new_QC_calls"] = 0
        out["updated_at"] = _now()
        sha = _write_json(INNER_VAL_JSON, out)
        _say("inner-val stage %s → %s (sha256=%s)" % (
            out["status"], INNER_VAL_JSON, sha[:16]))
    finally:
        _free_cuda()


def _dev_split_doc_split_path() -> Path:
    return Path(TIER_DEFAULTS[TIER]["tables_dir"]) / "split_manifest_v53mp.json"


def _run_dev(seeds: Sequence[int], device: str, cost: _Cost) -> None:
    if not INNER_VAL_JSON.is_file():
        raise RepairError("INNER_VAL_FIRST",
                          "run --splits inner-val before the dev stage")
    doc = _read_json(INNER_VAL_JSON)
    if doc.get("status") != "ok":
        raise RepairError(
            "INNER_VAL_GATES_FAILED",
            "inner-val stage status=%r — fail-closed, dev refused"
            % doc.get("status"))
    _assert_freeze_unchanged(doc.get("assets_frozen") or {})
    freeze = doc["assets_frozen"]
    if not SMOKE_REPORT.is_file():
        raise RepairError("SMOKE_FIRST", "smoke report missing")
    smoke = _read_json(SMOKE_REPORT)
    if not smoke.get("pass") or not smoke.get("forward_drift_probe"):
        raise RepairError("SMOKE_FAILED_OR_PROBE_MISSING",
                          "smoke must pass with a drift probe before dev")
    probe = smoke["forward_drift_probe"]
    if _sha256_file(SMOKE_REPORT) != doc.get("smoke_report_sha256"):
        raise RepairError("SMOKE_REPORT_DRIFTED",
                          "smoke_report changed since the inner-val stage")

    # production dev corpus: columnar + LAZY labels + frozen dev pack
    tables_dir = Path(TIER_DEFAULTS[TIER]["tables_dir"])
    split_path = _dev_split_doc_split_path()
    corpus = _build_tier(TIER, tables_dir, split_path, device,
                         int(TIER_DEFAULTS[TIER]["cache_curves"]), "auto")
    pack_info = corpus[5]
    if not pack_info.get("used"):
        raise RepairError(
            "DEV_PACK_UNAVAILABLE",
            "dev stage requires the frozen dev pack (low-memory path); got %s"
            % pack_info)
    dataset, labels, endpoint, dev_pos, dev_cache = (
        corpus[0], corpus[1], corpus[2], corpus[3], corpus[4])
    split_doc = json.loads(split_path.read_text(encoding="utf-8"))
    from scripts.campaigns.r3_v53mp_run import _subset_curve_ids
    dev_curves = sorted(_subset_curve_ids(split_doc, "dev"))
    from pes2ts_core.pesn.offline.data.key_atom_dev_pack_v53mp import (
        positions_sha256 as _positions_sha256,
    )
    dev_positions_sha = str(_positions_sha256(list(dev_pos)))
    n_dev = len(dev_pos)
    ok_mask = _ok_mask(dataset, dev_pos)
    n_ok = int(np.sum(ok_mask))
    _say("dev: positions=%d ok=%d curves=%d pos_sha=%s" % (
        n_dev, n_ok, len(dev_curves), dev_positions_sha[:16]))

    SEALED_DIR.mkdir(parents=True, exist_ok=True)
    ledger_entries: List[Dict[str, Any]] = []
    sealed_files: Dict[str, str] = {}
    per_seed: Dict[str, Any] = {}
    all_ok = True
    dev_corpus = {"dataset": dataset, "labels": labels, "endpoint": endpoint}

    def _seal(name: str, payload: Mapping[str, Any],
              model_name: str) -> None:
        path = SEALED_DIR / name
        sha = _write_json(path, payload)
        sealed_files[name] = sha
        entry = _ledger_entry(
            split="dev", model=model_name,
            output_path=str(path), output_sha256=sha,
            n_positions=n_dev, n_curves=len(dev_curves),
            curve_set_sha256=_canonical_curves_sha(dev_curves),
            positions_sha256=dev_positions_sha,
            code_sha256=_code_identity_sha(freeze),
            cost=cost.snapshot(device))
        _append_ledger(entry)
        ledger_entries.append(entry)
        _say("sealed %s (sha256=%s)" % (name, sha[:16]))

    # ---- C × seeds ------------------------------------------------------
    for seed in seeds:
        _say("=== dev C seed %d ===" % seed)
        hist_block = freeze["c_runs"][str(seed)]
        model, state = _load_c_model(seed, device)
        try:
            tau = ((state.get("tau_end_calibration") or {}).get("tau_end"))
            res = _eval_split(
                "dev", dev_pos, corpus=dev_corpus, cache=dev_cache,
                device=device, model=model, collect_process=True,
                tau_end=tau, cost=cost,
                ordering_q_noise=float(probe.get("max_abs_q_diff") or 0.0))
        finally:
            del model
            _free_cuda()

        fid_gated, fid_recorded = _dev_fidelity_gates(
            res["move"], res["process"], hist_block, probe,
            res.get("ordering_tie_tols"))
        den = _dev_denominator_gates(
            hist_block["history_denominators"], n_dev, n_ok,
            res["denominators"]["n_curves"],
            res["denominators"]["n_curves_move"], dev_positions_sha,
            hist_block.get("dev_pack_provenance"))
        seed_ok = (_pass_all(res["sanity_gates"]) and _pass_all(fid_gated)
                   and _pass_all(den))
        all_ok = all_ok and seed_ok
        per_seed[str(seed)] = {
            "checkpoint_sha256": hist_block["checkpoint"]["sha256"],
            "denominators": res["denominators"],
            "move": res["move"],
            "process": res["process"],
            "ordering_tie_tols": res.get("ordering_tie_tols"),
            "fidelity_gates": fid_gated,
            "fidelity_recorded": fid_recorded,
            "denominator_gates": den,
            "sanity_gates": res["sanity_gates"],
            "seed_pass": seed_ok,
        }
        _seal("v53mp_50k_C_s%d_dev.json" % seed,
              {"schema": SCHEMA_SEALED, "sealed": True,
               "split": "dev", "model": "C_s%d" % seed,
               "created_at": _now(),
               "checkpoint_sha256": hist_block["checkpoint"]["sha256"],
               "metrics_sha256": hist_block["metrics"]["sha256"],
               "denominators": res["denominators"],
               "move_before_after": res["move"],
               "process": res["process"],
               "ordering_tie_tols": res.get("ordering_tie_tols"),
               "fidelity_gates": fid_gated,
               "fidelity_recorded": fid_recorded,
               "denominator_gates": den,
               "sanity_gates": res["sanity_gates"],
               "seed_pass": seed_ok,
               "unseal_before_wp7": False},
              "C_s%d" % seed)
        _say("dev C seed %d: fidelity %d/%d pass, sanity %d/%d, denominators %d/%d" % (
            seed,
            sum(1 for g in fid_gated if g["pass"]), len(fid_gated),
            sum(1 for g in res["sanity_gates"] if g["pass"]),
            len(res["sanity_gates"]),
            sum(1 for g in den if g["pass"]), len(den)))
        del state
        _free_cuda()

    # ---- R3 reference (MOVE only, same dev) ------------------------------
    _say("=== dev R3 reference ===")
    r3_model = _load_r3_model(device)
    try:
        r3_res = _eval_split(
            "dev", dev_pos, corpus=dev_corpus, cache=dev_cache,
            device=device, model=r3_model, collect_process=False,
            tau_end=None, cost=cost)
    finally:
        del r3_model
        _free_cuda()
    r3_ok = _pass_all(r3_res["sanity_gates"])
    all_ok = all_ok and r3_ok
    _seal("r3_reference_dev.json",
          {"schema": SCHEMA_SEALED, "sealed": True, "split": "dev",
           "model": "r3_reference", "created_at": _now(),
           "checkpoint_sha256":
               freeze["r3_reference"]["checkpoint"]["sha256"],
           "metrics_sha256": freeze["r3_reference"]["metrics"]["sha256"],
           "denominators": r3_res["denominators"],
           "move_before_after": r3_res["move"],
           "sanity_gates": r3_res["sanity_gates"],
           "sanity_pass": r3_ok,
           "note": "H2 MOVE guard reference (R3 has no q/endpoint heads)",
           "unseal_before_wp7": False},
          "r3_reference")
    _say("dev R3: sanity %s | pairs %d→%d" % (
        "PASS" if r3_ok else "FAIL",
        r3_res["move"]["pair_denominators"]["n_pairs_collected_before"],
        r3_res["move"]["pair_denominators"]["n_pairs_collected_after"]))

    # ---- H2 guard table (dev, sealed) ------------------------------------
    h2_guards = []
    for seed in seeds:
        c_move = per_seed[str(seed)]["move"]
        for variant in ("after_repaired", "before_legacy"):
            guard = _h2_move_guard(
                c_move[variant], r3_res["move"][variant],
                split="dev", variant=variant, seed=seed)
            h2_guards.append(guard)
    h2_doc = {"schema": SCHEMA_SEALED, "sealed": True, "split": "dev",
              "created_at": _now(), "guards": h2_guards,
              "unseal_before_wp7": False}
    _seal("h2_move_guard_dev.json", h2_doc, "h2_move_guard")

    # ---- seal manifest ----------------------------------------------------
    manifest = {
        "schema": SCHEMA_SEAL_MANIFEST, "sealed": True,
        "created_at": _now(),
        "unseal_rule": ("WP1b baseline-repair access ONLY. No unseal and no "
                        "use for candidate design/threshold tuning before "
                        "WP7 (plan §4.3/§9.3). WP7 candidate evaluation is "
                        "a SEPARATE registered access."),
        "files": dict(sealed_files),
        "ledger_path": str(LEDGER_PATH),
        "ledger_entries_this_run": len(ledger_entries),
        "access": {"actor": _actor(), "purpose": "baseline_repair",
                   "wp": "WP1b", "split": "dev",
                   "curve_set_sha256": _canonical_curves_sha(dev_curves),
                   "positions_sha256": dev_positions_sha,
                   "code_sha256": _code_identity_sha(freeze),
                   "n_positions": n_dev, "n_curves": len(dev_curves)},
    }
    man_path = SEALED_DIR / "manifest.json"
    man_sha = _write_json(man_path, manifest)
    sealed_files["manifest.json"] = man_sha
    entry = _ledger_entry(
        split="dev", model="seal_manifest",
        output_path=str(man_path), output_sha256=man_sha,
        n_positions=n_dev, n_curves=len(dev_curves),
        curve_set_sha256=_canonical_curves_sha(dev_curves),
        positions_sha256=dev_positions_sha,
        code_sha256=_code_identity_sha(freeze),
        cost=cost.snapshot(device))
    _append_ledger(entry)
    ledger_entries.append(entry)

    # ---- update the readable audit with the SEALED status only -----------
    # (flags + denominator COUNTS only — no dev science values)
    doc["dev_sealed_status"] = {
        "status": "ok" if all_ok else "gates_failed",
        "pass": bool(all_ok),
        "sealed_dir": str(SEALED_DIR),
        "sealed_files": dict(sealed_files),
        "sealed_manifest_sha256": man_sha,
        "ledger_path": str(LEDGER_PATH),
        "ledger_entries_this_run": len(ledger_entries),
        "denominator_sanity": {
            "n_dev_positions": n_dev,
            "n_move_ok_positions": n_ok,
            "n_dev_curves": len(dev_curves),
            "n_dev_positions_expected": int(
                freeze["c_runs"][str(seeds[0])]["history_denominators"]
                .get("n_dev_samples") or -1),
            "n_move_ok_expected": int(
                freeze["c_runs"][str(seeds[0])]["history_denominators"]
                .get("move_n_val_samples") or -1),
            "n_pairs_before_total": int(sum(
                per_seed[str(s)]["move"]["pair_denominators"]
                ["n_pairs_collected_before"] for s in seeds)),
            "n_pairs_after_total": int(sum(
                per_seed[str(s)]["move"]["pair_denominators"]
                ["n_pairs_collected_after"] for s in seeds)),
            "n_ignored_non_key_total": int(sum(
                per_seed[str(s)]["move"]["pair_denominators"]
                ["pair_ownership_audit"].get("n_ignored_non_key_total", 0)
                for s in seeds)),
            "n_invalid_total": 0,
        },
        "fidelity_flags": {
            "C_s%d" % s: _gate_flags(per_seed[str(s)]["fidelity_gates"])
            for s in seeds},
        "fidelity_recorded_not_gated": {
            "C_s%d" % s: [str(g["name"])
                          for g in per_seed[str(s)]["fidelity_recorded"]]
            for s in seeds},
        "denominator_flags": {
            "C_s%d" % s: _gate_flags(per_seed[str(s)]["denominator_gates"])
            for s in seeds},
        "sanity_flags": {
            "C_s%d" % s: _gate_flags(per_seed[str(s)]["sanity_gates"])
            for s in seeds},
        "r3_sanity_flags": _gate_flags(r3_res["sanity_gates"]),
        "forward_drift_probe": {
            "max_abs_u_diff": probe.get("max_abs_u_diff"),
            "u_composition_invariant": probe.get("u_composition_invariant"),
            "dead_hist_tol": probe.get("dead_hist_tol"),
        },
        "note": ("flags are pass/fail + denominator COUNTS only — no "
                 "fine-grained dev science metrics are exposed here"),
        "cost": cost.snapshot(device),
        "updated_at": _now(),
    }
    doc.setdefault("cost", {})["dev_stage"] = cost.snapshot(device)
    if not all_ok:
        doc["status"] = "dev_gates_failed"
    # (all_ok: inner-val status was verified "ok" at stage entry → keep it)
    doc["updated_at"] = _now()
    _write_json(INNER_VAL_JSON, doc)
    _say("dev stage %s; sealed %d files; ledger +%d entries" % (
        doc["dev_sealed_status"]["status"], len(sealed_files),
        len(ledger_entries)))


# ---------------------------------------------------------------------------
# ledger
# ---------------------------------------------------------------------------


def _ledger_entry(*, split: str, model: str, output_path: str,
                  output_sha256: str, n_positions: int, n_curves: int,
                  curve_set_sha256: str, positions_sha256: str,
                  code_sha256: str, cost: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "schema": SCHEMA_LEDGER,
        "entry_id": "%s_%s_%d" % (split, model, int(time.time() * 1000)),
        "actor": _actor(),
        "purpose": "baseline_repair",
        "wp": "WP1b",
        "campaign": "V5.4.1",
        "split": split,
        "model": model,
        "ts": _now(),
        "curve_set_sha256": curve_set_sha256,
        "positions_sha256": positions_sha256,
        "n_positions": int(n_positions),
        "n_curves": int(n_curves),
        "code_sha256": code_sha256,
        "output_path": output_path,
        "output_sha256": output_sha256,
        "sealed": split == "dev",
        "unseal_before_wp7": False,
        "new_QC_calls": 0,
        "cost": dict(cost),
        "status": "done",
    }


def _append_ledger(entry: Mapping[str, Any]) -> None:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    with open(LEDGER_PATH, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(_jsonable(entry), sort_keys=True,
                            ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


# ---------------------------------------------------------------------------
# stage: verify (integrity re-check; no science values printed)
# ---------------------------------------------------------------------------


def cmd_verify(_args: argparse.Namespace) -> int:
    report: Dict[str, Any] = {"schema": SCHEMA_VERIFY,
                              "created_at": _now(), "checks": {},
                              "new_QC_calls": 0}
    checks = report["checks"]
    ok = True

    def _check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        checks[name] = {"pass": bool(cond), "detail": detail}
        ok = ok and bool(cond)
        _say("VERIFY %s: %s%s" % (name, "PASS" if cond else "FAIL",
                                  (" — " + detail) if detail else ""))

    try:
        doc = _read_json(INNER_VAL_JSON)
        _check("inner_val_status_ok", doc.get("status") == "ok",
               "status=%s" % doc.get("status"))
        dss = doc.get("dev_sealed_status") or {}
        _check("dev_sealed_status_ok", dss.get("status") == "ok",
               "status=%s" % dss.get("status"))
        try:
            _assert_freeze_unchanged(doc.get("assets_frozen") or {})
            _check("freeze_unchanged", True, "code+assets match freeze")
        except RepairError as exc:
            _check("freeze_unchanged", False, str(exc))
        if SMOKE_REPORT.is_file():
            _check("smoke_report_unchanged",
                   _sha256_file(SMOKE_REPORT)
                   == doc.get("smoke_report_sha256"),
                   "smoke_report matches the inner-val stage record")
            sm = _read_json(SMOKE_REPORT)
            _check("smoke_passed", bool(sm.get("pass")), "smoke_report")
        else:
            _check("smoke_report_exists", False, str(SMOKE_REPORT))
        man_path = SEALED_DIR / "manifest.json"
        _check("seal_manifest_exists", man_path.is_file(), str(man_path))
        if man_path.is_file():
            man = _read_json(man_path)
            bad = []
            for name, sha in (man.get("files") or {}).items():
                p = SEALED_DIR / name
                if not p.is_file() or _sha256_file(p) != sha:
                    bad.append(name)
            _check("sealed_files_sha_match", not bad,
                   ("mismatched: %s" % bad) if bad else
                   "%d files" % len(man.get("files") or {}))
            _check("seal_manifest_self_sha",
                   _sha256_file(man_path)
                   == dss.get("sealed_manifest_sha256"),
                   "manifest sha recorded in dev_sealed_status")
            _check("seal_rule_recorded",
                   "WP7" in str(man.get("unseal_rule", "")),
                   "unseal rule present")
        if LEDGER_PATH.is_file():
            entries = [json.loads(l) for l in
                       LEDGER_PATH.read_text(encoding="utf-8").splitlines()
                       if l.strip()]
            # append-only ledger: a re-run re-writes sealed files, so the
            # LAST entry per output_path is the access that owns the bytes
            # currently on disk; earlier entries remain as history.
            last_by_path: Dict[str, Dict[str, Any]] = {}
            for e in entries:
                last_by_path[str(e.get("output_path"))] = e
            bad = []
            for path_str, e in last_by_path.items():
                p = Path(path_str)
                if not p.is_file() or _sha256_file(p) != e.get("output_sha256"):
                    bad.append(e.get("entry_id"))
            _check("ledger_output_shas_match", not bad,
                   ("mismatched: %s" % bad) if bad else
                   "%d entries (%d distinct outputs)" % (
                       len(entries), len(last_by_path)))
            _check("ledger_purpose_pinned",
                   all(e.get("purpose") == "baseline_repair"
                       for e in entries),
                   "all entries purpose=baseline_repair")
            _check("ledger_no_unseal",
                   all(e.get("unseal_before_wp7") is False for e in entries),
                   "no entry unseals before WP7")
            report["ledger_entries"] = len(entries)
        else:
            _check("ledger_exists", False, str(LEDGER_PATH))
        cg = []
        if CGROUP_DIR.is_dir():
            for p in sorted(CGROUP_DIR.glob("*/cgroup_summary.json")):
                s = _read_json(p)
                cg.append({"stage_dir": str(p.parent),
                           "peak_anon_bytes": s.get("peak_anon_bytes"),
                           "oom_kill": s.get("oom_kill"),
                           "gate_exit": s.get("gate_exit"),
                           "max_swap_current_bytes":
                               s.get("max_swap_current_bytes")})
        report["cgroup_evidence"] = cg
        _check("cgroup_gates_clean",
               bool(cg) and all(int(x.get("oom_kill") or 0) == 0
                                and int(x.get("gate_exit") or 0) == 0
                                for x in cg),
               "%d stage summaries" % len(cg))
        report["cost"] = doc.get("cost", {})
        report["dev_sealed_status_summary"] = {
            "status": dss.get("status"),
            "pass": dss.get("pass"),
            "denominator_sanity": dss.get("denominator_sanity"),
        }
    except Exception as exc:
        _check("verify_completed", False,
               "%s: %s" % (type(exc).__name__, exc))
        report["failure"] = {"error": str(exc),
                             "traceback": traceback.format_exc()[-4000:]}
        ok = False

    report["pass"] = bool(ok)
    sha = _write_json(VERIFY_REPORT, report)
    _say("verify %s → %s (sha256=%s)" % (
        "PASS" if ok else "FAIL", VERIFY_REPORT, sha[:16]))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# failure recorder + main
# ---------------------------------------------------------------------------


def _record_failure(stage: str, exc: BaseException) -> Path:
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    path = AUDIT_DIR / ("failure_%s_%s.json" % (stage, ts))
    doc = {
        "schema": SCHEMA_FAILURE,
        "stage": stage,
        "created_at": _now(),
        "actor": _actor(),
        "purpose": "baseline_repair",
        "status": "failed",
        "error_code": getattr(exc, "code", type(exc).__name__),
        "error": "%s: %s" % (type(exc).__name__, exc),
        "traceback": traceback.format_exc()[-6000:],
        "new_QC_calls": 0,
        "fail_closed": True,
        "r3_guard": ("NOT skipped — a failed R3 re-eval keeps T1 blocked "
                     "(plan §4.3)"),
    }
    sha = _write_json(path, doc)
    _say("FAILURE recorded at %s (sha256=%s)" % (path, sha[:16]))
    if stage == "dev":
        try:
            _append_ledger({
                "schema": SCHEMA_LEDGER,
                "entry_id": "dev_failure_%d" % int(time.time() * 1000),
                "actor": _actor(), "purpose": "baseline_repair",
                "wp": "WP1b", "campaign": "V5.4.1", "split": "dev",
                "model": "stage_failure", "ts": _now(),
                "error_code": getattr(exc, "code", type(exc).__name__),
                "output_path": str(path), "output_sha256": sha,
                "sealed": True, "unseal_before_wp7": False,
                "new_QC_calls": 0, "status": "failed",
            })
        except Exception:  # pragma: no cover — ledger must not mask failure
            pass
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="v541_baseline_repair",
        description="V5.4.1 WP1b baseline repair re-evaluation (§4.3)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("smoke", help="fixtures + one inner-val position vs prod")
    p_run = sub.add_parser("run", help="execute a baseline-repair stage")
    p_run.add_argument("--splits", choices=("inner-val", "dev"),
                       required=True)
    p_run.add_argument("--seeds",
                       default=",".join(str(s) for s in SEEDS_DEFAULT))
    p_run.add_argument("--device", default="cuda")
    sub.add_parser("verify", help="integrity re-check of all outputs")
    args = ap.parse_args(argv)

    seeds = (tuple(int(s) for s in str(args.seeds).split(",") if s.strip())
             if args.cmd == "run" else SEEDS_DEFAULT)
    stage = args.cmd if args.cmd != "run" else "run_%s" % args.splits
    # all paths in this module are repo-root relative (single convention)
    os.chdir(str(PROJECT_ROOT))

    if args.cmd == "smoke":
        try:
            return cmd_smoke(args)
        except Exception as exc:
            _record_failure("smoke", exc)
            _say("smoke FAILED: %s" % exc)
            return 1

    if args.cmd == "verify":
        try:
            return cmd_verify(args)
        except Exception as exc:
            _record_failure("verify", exc)
            _say("verify FAILED: %s" % exc)
            return 1

    cost = _Cost()
    lock = _GpuLock()
    try:
        lock.acquire()
    except RepairError as exc:
        _record_failure(stage, exc)
        _say(str(exc))
        return 1
    try:
        if args.splits == "inner-val":
            _run_inner_val(seeds, args.device, cost)
        else:
            _run_dev(seeds, args.device, cost)
    except Exception as exc:
        _record_failure(stage, exc)
        _say("stage %s FAILED: %s" % (stage, exc))
        return 1
    finally:
        lock.release()
    doc = _read_json(INNER_VAL_JSON)
    status = (doc.get("dev_sealed_status") or {}).get("status") \
        if args.splits == "dev" else doc.get("status")
    _say("stage %s done status=%s" % (stage, status))
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
