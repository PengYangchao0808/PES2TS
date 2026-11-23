#!/usr/bin/env python3
"""Build the frame-level master key crosswalk for the PES2TS datastore.

Deliverables (default ``--out-dir /pes2ts_datastore/01_master_index``)
--------------------------------------------------------------------
``frame_crosswalk.parquet``
    One row per canonical V5 ``idx_v2`` frame: ``curves.json`` curve order,
    dense ``node_order`` expansion => 7,342,581 rows.  Welds the canonical
    frame to the matched xTB single-point row (``xtb_frame_key``/``has_xtb``)
    or records an explicit ``excluded_reason``.  ``global_frame_id`` is the
    repository-wide unique frame row number.
``frame_crosswalk_manifest.json``
    Build manifest: source sha256s, join semantics + evidence, coverage and
    reconciliation counters, exclusion accounting, acceptance checks,
    idempotency hashes, boundaries/limitations.
``xtb_row_ledger.parquet``
    All 2,877,830 xTB manifest rows with disposition
    (``joined`` / ``excluded`` / ``unmatched``) and reason -- no row is
    silently dropped.
``excluded_ledger.parquet``
    Explicit reaction-level (dft_energy_unavailable / unsupported_layout /
    duplicate_frame_key) and branch-level (short_branch) exclusion records
    with point accounting.

Join semantics (derived from ``curves.json`` + ``label_curves/node_order.npy``
+ ``geom/curve_offsets.json`` + the xTB manifest rows; see
``join_semantics`` in the manifest for the full evidence):

* ``node_order`` maps **canonical path position -> idx_v2 node index**.  It is
  identity for all 64,195 curves of this delivery (all
  ``endpoint_a_index == 0``), hence
  ``canonical_node_index == idx_v2_node_index``.
* ``node_order`` does **not** encode the source IRC trajectory frame number.
  The xTB ``frame_index`` is the *source h5 trajectory frame index* (TS-first
  layout: frame 0 is the TS).  For one reaction, with
  ``ts = curves[i].ts_index`` (canonical path position of the TS),
  ``n = curves[i].n_nodes`` and ``p = branch_position``:

  * orientation ``A_increasing``:
    A ``p`` -> canonical ``ts + p``; B ``p>=1`` -> canonical ``ts - p``;
  * orientation ``A_decreasing``:
    A ``p`` -> canonical ``ts - p``; B ``p>=1`` -> canonical ``ts + p``;
  * direction B ``p == 0`` is the ``near_ts_replica`` frame: it duplicates the
    canonical TS node with a *different* near-TS geometry, so it has no
    independent canonical frame (ledger reason ``duplicate_frame_key``);
  * source/DFT frame number of a canonical position:
    ``A_increasing``: ``p`` on the A side, ``n - pos`` on the B side;
    ``A_decreasing``: ``ts - pos`` on the A side, ``pos + 1`` on the B side.
    This is written to ``dft_frame_index`` and cross-checked against every
    joined xTB ``frame_index`` (must agree 100%).

The source trajectory has ``n_frames = n_nodes + 1`` frames: the extra frame
is exactly the excluded near-TS replica.  Orientation is resolved per
reaction by exact round-10 geometry equality of all sampled rows, so the
2 ``A_decreasing`` reactions in this delivery are handled without guessing.

Read-only on every source; writes only ``--out-dir`` (plus an optional log
under ``/tmp``).  Usage::

    wsl --cd /mnt/e/.../PES2TS -e bash infra/runtime/wsl_guarded_run.sh \
        --mem 6G --name build-crosswalk --outdir /tmp/cw-cgroup -- \
        /root/miniforge3/envs/pes2ts-gpu/bin/python \
        pes2ts_data/builders/build_frame_crosswalk.py --stage all
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"pyarrow is required: {exc}") from exc


CROSSWALK_SCHEMA = "pes2ts_frame_crosswalk_v1"
MANIFEST_SCHEMA = "pes2ts_frame_crosswalk_manifest_v1"
LEDGER_SCHEMA = "pes2ts_xtb_row_ledger_v1"
EXCLUDED_SCHEMA = "pes2ts_frame_exclusion_ledger_v1"
IDX_SCHEMA = "v53mp_idx_v2"

CURVES_JSON = "idx_v2/curves.json"
NODE_ORDER = "idx_v2/label_curves/node_order.npy"
OFFSETS = "idx_v2/label_curves/offsets.npy"
GEOM_OFFSETS = "geom/curve_offsets.json"
GEOM_COORDS = "geom/coords.f64.npy"
DATA_MANIFEST = "data_manifest.json"
SPLIT_MANIFEST = "split_manifest_v53mp.json"

XTB_INDEX = "manifest/index.json"
XTB_SHARDS = "manifest/shards/shard_*.jsonl"
XTB_FAILURES = "run/failures/shard_*.jsonl"
XTB_AUDIT = "run/audit.json"
XTB_REGISTRY = "data/manifests/xtb_singlepoints_train_fit_v1.json"

ORIENT_UNKNOWN, ORIENT_INC, ORIENT_DEC = 0, 1, 2
DISP_JOINED, DISP_EXCLUDED, DISP_UNMATCHED = 1, 2, 3
R_NONE, R_SCF, R_REPLICA, R_NOT_IN_IDX = 0, 1, 2, 3
REASON_NAMES = {R_NONE: "", R_SCF: "scf_not_converged",
                R_REPLICA: "duplicate_frame_key",
                R_NOT_IN_IDX: "reaction_not_in_idx_v2"}
DISP_NAMES = {DISP_JOINED: "joined", DISP_EXCLUDED: "excluded",
              DISP_UNMATCHED: "unmatched"}
FRAME_REASON_VALUES = ("", "scf_not_converged", "short_branch",
                       "unsupported_layout", "dft_energy_unavailable",
                       "duplicate_frame_key")
CROSSWALK_CHUNK_ROWS = 200_000


class BuildError(RuntimeError):
    """Typed fail-closed build failure (manifest is written with FAIL state)."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path | str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def sha_or_none(path: Path) -> str | None:
    return sha256_file(path) if path.exists() else None


def round10(a: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(a, dtype=np.float64), 10)


def coords_sha256(frame: np.ndarray) -> str:
    """Frozen xTB fingerprint recipe, used for frames without an xTB row.

    Reverse-engineered from the xTB manifest and verified against every
    joined row::

        sha256(json.dumps(np.round(coords, 10).tolist(),
                          separators=(",", ":")).encode())
    """
    payload = json.dumps(round10(frame).tolist(), separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def log_line(msg: str, t0: float, fh=None) -> None:
    line = f"[crosswalk +{time.perf_counter() - t0:8.1f}s] {msg}"
    print(line, file=sys.stderr, flush=True)
    if fh is not None:
        fh.write(line + "\n")
        fh.flush()


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
class CrosswalkBuild:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.idx_root = Path(args.idx_root)
        self.xtb_root = Path(args.xtb_root)
        self.out_dir = Path(args.out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = args.log
        self.log_fh = open(self.log_path, "a", encoding="utf-8") if self.log_path else None
        self.t0 = time.perf_counter()
        self.rows = 0
        self.warnings: list[str] = []

    # -- logging ------------------------------------------------------------
    def log(self, msg: str) -> None:
        log_line(msg, self.t0, self.log_fh)

    def close(self) -> None:
        if self.log_fh:
            self.log_fh.close()

    # -- phase 0 ------------------------------------------------------------
    def load_sources(self) -> dict:
        log = self.log
        log("[0/5] loading frozen idx_v2 index, node_order/offsets and geometry pack")
        self.curves_doc = load_json(self.idx_root / CURVES_JSON)
        if self.curves_doc.get("schema") != IDX_SCHEMA:
            raise BuildError(f"curves.json schema "
                             f"{self.curves_doc.get('schema')!r} != {IDX_SCHEMA!r}")
        self.binding = self.curves_doc.get("binding", {})
        self.curve_ids = list(self.curves_doc["curve_ids"])
        self.curve_index = {c: i for i, c in enumerate(self.curve_ids)}
        self.curves = self.curves_doc["curves"]
        if len(self.curves) != len(self.curve_ids):
            raise BuildError("curves.json curves/curve_ids length mismatch")
        if int(self.binding.get("n_curves", -1)) != len(self.curve_ids):
            raise BuildError("curves.json binding n_curves mismatch")
        self.n_nodes = np.array([int(c["n_nodes"]) for c in self.curves],
                                dtype=np.int64)
        self.ts_index = np.array([int(c["ts_index"]) for c in self.curves],
                                 dtype=np.int64)
        self.n_atoms = np.array([len(c.get("atomic_numbers") or ())
                                 for c in self.curves], dtype=np.int64)
        self.endpoint_a = np.array([int(c["endpoint_a"]) for c in self.curves],
                                   dtype=np.int64)
        self.endpoint_b = np.array([int(c["endpoint_b"]) for c in self.curves],
                                   dtype=np.int64)

        self.offsets = np.load(self.idx_root / OFFSETS, mmap_mode="r")
        self.node_order = np.load(self.idx_root / NODE_ORDER, mmap_mode="r")
        self.n_frames_total = int(len(self.node_order))
        if int(self.offsets[-1]) != self.n_frames_total:
            raise BuildError("offsets[-1] != node_order length")
        if int(self.n_nodes.sum()) != self.n_frames_total:
            raise BuildError("sum(curves.n_nodes) != node_order length")

        # node_order semantics
        ident = 0
        for i in range(len(self.curve_ids)):
            lo, hi = int(self.offsets[i]), int(self.offsets[i + 1])
            k = hi - lo
            if np.array_equal(np.asarray(self.node_order[lo:hi]), np.arange(k)):
                ident += 1
        self.n_identity_curves = ident
        self.n_nonidentity_curves = len(self.curve_ids) - ident
        if self.n_nonidentity_curves:
            log(f"    WARNING: {self.n_nonidentity_curves} curves use a "
                "non-identity node_order; the general mapping is applied")
        else:
            log(f"    node_order maps canonical position -> idx_v2 node index; "
                f"identity for all {ident} curves")
        limit = int(getattr(self.args, "smoke_max_curves", 0) or 0)
        self.limit_curves = min(len(self.curve_ids), limit) if limit else \
            len(self.curve_ids)
        self.expected_frames = int(self.n_nodes[:self.limit_curves].sum())
        if self.limit_curves != len(self.curve_ids):
            log(f"    SMOKE MODE: only the first {self.limit_curves} curves / "
                f"{self.expected_frames} frames are built")

        geom_doc = load_json(self.idx_root / GEOM_OFFSETS)
        if geom_doc.get("schema") != "v53mp_geom_pack_v1":
            raise BuildError("unexpected geometry pack schema")
        self.geom_offsets = geom_doc["curves"]
        self.geom_binding = geom_doc.get("binding", {})
        self.coords_mm = np.load(self.idx_root / GEOM_COORDS, mmap_mode="r")
        if int(self.geom_binding.get("n_curves", -1)) != len(self.curve_ids):
            raise BuildError("geometry pack curve count mismatch")
        for i, cid in enumerate(self.curve_ids):
            g = self.geom_offsets.get(cid)
            if g is None:
                raise BuildError(f"geometry pack missing curve {cid}")
            if int(g["n_nodes"]) != int(self.n_nodes[i]):
                raise BuildError(f"geometry n_nodes mismatch for {cid}")
            if int(g["n_atoms"]) != int(self.n_atoms[i]):
                raise BuildError(f"geometry n_atoms mismatch for {cid}")
        log(f"    {len(self.curve_ids)} curves / {self.n_frames_total} canonical "
            f"frames / {self.coords_mm.shape[0]} geom atom rows")

        log("[0/5] loading xTB manifest index + failure ledger")
        self.xtb_index = load_json(self.xtb_root / XTB_INDEX)
        idx = self.xtb_index
        for key in ("dft_energy_unavailable_ids", "unsupported_layout_ids",
                    "duplicate_frame_key_ids", "invalid_payload_ids",
                    "missing_inventory_ids"):
            if key not in idx:
                raise BuildError(f"xTB index.json missing {key}")
        self.xtb_sets = {
            "dft_energy_unavailable": set(idx["dft_energy_unavailable_ids"]),
            "unsupported_layout": set(idx["unsupported_layout_ids"]),
            "duplicate_frame_key": set(idx["duplicate_frame_key_ids"]),
            "invalid_payload": set(idx["invalid_payload_ids"]),
            "missing_inventory": set(idx["missing_inventory_ids"]),
        }
        self.xtb_reasons = idx.get("dft_energy_unavailable_reasons") or {}
        failures: dict[str, dict] = {}
        fail_files = sorted(glob.glob(str(self.xtb_root / XTB_FAILURES)))
        for path in fail_files:
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("schema_version") != "pes2ts_xtb_failure_v1":
                        raise BuildError(f"{path}: bad failure schema")
                    key = row["frame_key"]
                    if key in failures:
                        raise BuildError(f"duplicate failure frame_key {key}")
                    if row.get("reason") != "scf_not_converged":
                        raise BuildError("unexpected failure reason "
                                         f"{row.get('reason')!r}")
                    failures[key] = row
        self.failures = failures
        self.n_failure_rows = len(failures)
        audit = None
        if (self.xtb_root / XTB_AUDIT).exists():
            audit = load_json(self.xtb_root / XTB_AUDIT)
            if not audit.get("passed", False):
                raise BuildError("xTB run/audit.json passed != true")
            n_fail_expected = int(audit.get("counts", {}).get("n_failure", -1))
            if n_fail_expected != self.n_failure_rows:
                raise BuildError(f"failure ledger {self.n_failure_rows} != "
                                 f"audit n_failure {n_fail_expected}")
        self.xtb_audit = audit
        log(f"    xTB: {idx.get('n_manifest_rows')} manifest rows / "
            f"{idx.get('n_reactions')} reactions / "
            f"{self.n_failure_rows} failure rows")

        # shortfall accounting from the frozen xTB index (must self-agree)
        sf = {
            "short_branch": int(idx.get("shortfall_points_short_branch", -1)),
            "dft_energy_unavailable": int(
                idx.get("shortfall_points_dft_energy_unavailable", -1)),
            "unsupported_layout": int(
                idx.get("shortfall_points_unsupported_layout", -1)),
            "duplicate_frame_key": int(
                idx.get("shortfall_points_duplicate_frame_key", -1)),
        }
        if sum(sf.values()) != int(idx.get("shortfall_points", -1)):
            raise BuildError("xTB shortfall bucket sum != shortfall_points")
        self.xtb_shortfall = sf
        self.xtb_nominal = int(idx.get("nominal_points", -1))
        self.xtb_actual = int(idx.get("n_manifest_rows", -1))
        if self.xtb_nominal - self.xtb_actual != int(idx["shortfall_points"]):
            raise BuildError("nominal - actual != shortfall_points")

        # split membership cross-check (dev has no xTB coverage by design)
        split_path = self.idx_root / SPLIT_MANIFEST
        self.split_subsets = {}
        if split_path.exists():
            split_doc = load_json(split_path)
            subsets = split_doc.get("subsets") or {}
            self.split_subsets = {
                "dev": set(subsets.get("dev") or []),
                "confirm": set(subsets.get("confirm") or []),
                "pool_50k": set((subsets.get("pool") or {}).get("50k") or []),
            }
        return {}

    def collect_hashes(self) -> dict:
        log = self.log
        log("[0/5] computing source sha256s")
        repo_root = Path(__file__).resolve().parents[2]
        registry = repo_root / XTB_REGISTRY
        out: dict = {
            "idx_v2_root": str(self.idx_root),
            "xtb_root": str(self.xtb_root),
            "curves_json_sha256": sha256_file(self.idx_root / CURVES_JSON),
            "node_order_sha256": sha256_file(self.idx_root / NODE_ORDER),
            "offsets_sha256": sha256_file(self.idx_root / OFFSETS),
            "geom_offsets_sha256": sha256_file(self.idx_root / GEOM_OFFSETS),
            "geom_coords_sha256": sha256_file(self.idx_root / GEOM_COORDS),
            "data_manifest_sha256_binding": self.binding.get(
                "data_manifest_sha256"),
            "split_manifest_sha256_binding": self.binding.get(
                "split_manifest_sha256"),
        }
        dm = self.idx_root / DATA_MANIFEST
        if dm.exists():
            out["data_manifest_sha256_file"] = sha256_file(dm)
            if out.get("data_manifest_sha256_binding") and \
                    out["data_manifest_sha256_file"] != out["data_manifest_sha256_binding"]:
                raise BuildError("data_manifest.json sha256 != curves binding")
        sm = self.idx_root / SPLIT_MANIFEST
        if sm.exists():
            out["split_manifest_sha256_file"] = sha256_file(sm)
            if out.get("split_manifest_sha256_binding") and \
                    out["split_manifest_sha256_file"] != out["split_manifest_sha256_binding"]:
                raise BuildError("split_manifest sha256 != curves binding")
        if not registry.exists():
            raise BuildError(f"xTB registry not found: {registry}")
        out["xtb_registry_path"] = str(registry)
        out["xtb_registry_sha256"] = sha256_file(registry)
        for name, rel in (("xtb_index_json_sha256", XTB_INDEX),
                          ("xtb_audit_json_sha256", XTB_AUDIT)):
            p = self.xtb_root / rel
            if p.exists():
                out[name] = sha256_file(p)
        fail_files = sorted(glob.glob(str(self.xtb_root / XTB_FAILURES)))
        h = hashlib.sha256()
        for p in fail_files:
            h.update(Path(p).read_bytes())
        out["xtb_failures_ledger_sha256"] = h.hexdigest()
        if self.geom_binding.get("coords_sha256") and \
                out["geom_coords_sha256"] != self.geom_binding["coords_sha256"]:
            raise BuildError("geom coords sha256 != geom binding")
        out["idx_v2_binding"] = self.binding
        return out

    # -- phase 1: index xTB rows + validate join ----------------------------
    def index_xtb_rows(self) -> None:
        log = self.log
        idx = self.xtb_index
        N = int(idx["n_manifest_rows"])
        shards = idx.get("shards")
        if not isinstance(shards, list) or not shards:
            raise BuildError("xTB index.json has no shard list")
        shard_paths = sorted(glob.glob(str(self.xtb_root / XTB_SHARDS)))
        if len(shard_paths) != len(shards):
            raise BuildError(f"manifest shard count {len(shard_paths)} != "
                             f"index {len(shards)}")
        exp = {str(s["name"]): int(s["n_rows"]) for s in shards}
        max_shards = self.args.smoke_max_shards
        if max_shards:
            shard_paths = shard_paths[:max_shards]
            N = sum(exp[Path(p).name] for p in shard_paths)
            log(f"[1/5] SMOKE: indexing {N} rows from {len(shard_paths)} shards")
        else:
            log(f"[1/5] indexing {N} xTB manifest rows from {len(shard_paths)} shards")

        self.row_dir = np.zeros(N, dtype=np.uint8)
        self.row_p = np.zeros(N, dtype=np.int16)
        self.row_fi = np.zeros(N, dtype=np.int32)
        self.row_nframes = np.zeros(N, dtype=np.int32)
        self.row_blen = np.zeros(N, dtype=np.int32)
        self.row_short = np.zeros(N, dtype=np.uint8)
        self.row_failed = np.zeros(N, dtype=np.uint8)
        self.row_key = np.zeros((N, 64), dtype=np.uint8)
        self.row_csha = np.zeros((N, 64), dtype=np.uint8)
        self.row_disp = np.zeros(N, dtype=np.uint8)
        self.row_reason = np.zeros(N, dtype=np.uint8)
        self.row_curve = np.full(N, -1, dtype=np.int32)
        self.row_canon = np.full(N, -1, dtype=np.int32)
        self.row_maxdiff = np.full(N, np.nan, dtype=np.float32)
        self.row_gchecked = np.zeros(N, dtype=np.uint8)
        self.row_global = np.full(N, -1, dtype=np.int64)
        n_curves = len(self.curve_ids)
        self.row_start_by_curve = np.full(n_curves, -1, dtype=np.int64)
        self.row_end_by_curve = np.full(n_curves, -1, dtype=np.int64)
        self.orient = np.zeros(n_curves, dtype=np.int8)
        self.short_a = np.zeros(n_curves, dtype=bool)
        self.short_b = np.zeros(n_curves, dtype=bool)
        self.branch_a_len = np.zeros(n_curves, dtype=np.int32)
        self.branch_b_len = np.zeros(n_curves, dtype=np.int32)

        state = {
            "row": 0, "group_start": 0, "group_rid": None, "group_rows": [],
            "n_replica": 0, "n_joined": 0, "n_failed_mapped": 0,
            "n_geom_checked": 0, "n_geom_mismatch": 0, "geom_maxdiff": 0.0,
            "n_inc": 0, "n_dec": 0, "orient_fail": [],
        }
        rid_seen: dict[str, int] = {}
        group_bounds: list[tuple[int, int]] = []
        group_reactions: list[str] = []
        counts_per_shard: dict[str, int] = {}

        def flush_group() -> None:
            rid = state["group_rid"]
            if rid is None:
                return
            rows = state["group_rows"]
            gstart = state["group_start"]
            gend = state["row"]
            group_bounds.append((gstart, gend))
            group_reactions.append(rid)
            self._validate_and_classify_group(rid, rows, gstart, gend, state)
            state["group_rows"] = []
            state["group_rid"] = None

        t0 = time.perf_counter()
        for shard_i, path in enumerate(shard_paths):
            name = Path(path).name
            if name not in exp:
                raise BuildError(f"shard {name} absent from index.json")
            n_shard = 0
            with open(path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    r = json.loads(line)
                    if r.get("schema_version") != "pes2ts_xtb_frame_v1":
                        raise BuildError(f"{name}: bad schema_version "
                                         f"{r.get('schema_version')!r}")
                    rid = r["reaction_id"]
                    if rid != state["group_rid"]:
                        flush_group()
                        if rid in rid_seen:
                            raise BuildError(
                                f"reaction {rid} reappears after other reactions "
                                "(rows are not contiguous in manifest order)")
                        rid_seen[rid] = 1
                        state["group_rid"] = rid
                        state["group_start"] = state["row"]
                    i = state["row"]
                    if i >= N:
                        raise BuildError("more manifest rows than index declares")
                    key = r["frame_key"]
                    if len(key) != 64:
                        raise BuildError(f"{name}: bad frame_key {key!r}")
                    self.row_dir[i] = 0 if r["direction"] == "A" else 1
                    self.row_p[i] = int(r["branch_position"])
                    self.row_fi[i] = int(r["frame_index"])
                    self.row_nframes[i] = int(r["n_frames"])
                    self.row_blen[i] = int(r["branch_length"])
                    self.row_short[i] = 1 if r["short_branch"] else 0
                    failed = key in self.failures
                    self.row_failed[i] = 1 if failed else 0
                    self.row_key[i] = np.frombuffer(key.encode("ascii"),
                                                    dtype=np.uint8)
                    self.row_csha[i] = np.frombuffer(
                        r["coords_sha256"].encode("ascii"), dtype=np.uint8)
                    state["group_rows"].append(r)
                    state["row"] += 1
                    n_shard += 1
            if n_shard != exp[name]:
                raise BuildError(f"{name}: n_rows {n_shard} != index {exp[name]}")
            counts_per_shard[name] = n_shard
            if (shard_i + 1) % 20 == 0:
                log(f"    {shard_i + 1}/{len(shard_paths)} shards, "
                    f"{state['row']} rows, {len(rid_seen)} reactions "
                    f"({time.perf_counter() - t0:.0f}s)")
        flush_group()
        row = state["row"]
        if row != N:
            raise BuildError(f"indexed {row} rows != xTB index {N}")
        if not max_shards and len(rid_seen) != int(idx["n_reactions"]):
            raise BuildError(f"{len(rid_seen)} reactions != "
                             f"{idx['n_reactions']}")
        if not max_shards:
            keys = self.row_key.view("S64").ravel()
            if len(np.unique(keys)) != len(keys):
                raise BuildError("duplicate xTB frame_key across manifest rows")
        log(f"    indexed {row} rows / {len(rid_seen)} reactions; joined="
            f"{state['n_joined']}, replica={state['n_replica']}, "
            f"failed_mapped={state['n_failed_mapped']}; orientation inc="
            f"{state['n_inc']} dec={state['n_dec']} "
            f"({time.perf_counter() - t0:.0f}s)")
        if state["n_geom_mismatch"]:
            raise BuildError(f"{state['n_geom_mismatch']} mapped rows fail "
                             "round10 geometry equality")
        if state["orient_fail"]:
            raise BuildError(f"{len(state['orient_fail'])} reactions have no "
                             f"unique geometry-resolved orientation: "
                             f"{state['orient_fail'][:3]}")
        self.rows = row
        self.group_bounds = group_bounds
        self.group_reactions = group_reactions
        self.counts_per_shard = counts_per_shard
        self.n_replica = state["n_replica"]
        self.n_mapped_joined = state["n_joined"]
        self.n_mapped_failed = state["n_failed_mapped"]
        self.n_geom_checked = state["n_geom_checked"]
        self.n_geom_mismatch = state["n_geom_mismatch"]
        self.geom_maxdiff = state["geom_maxdiff"]
        self.n_orient_inc = state["n_inc"]
        self.n_orient_dec = state["n_dec"]
        self.unique_reactions = len(rid_seen)

    def _validate_and_classify_group(self, rid, rows, gstart, gend, state) -> None:
        self._structural_checks(rid, rows)
        by_dir: dict[str, list[dict]] = {"A": [], "B": []}
        for r in rows:
            by_dir[r["direction"]].append(r)
        for d in ("A", "B"):
            by_dir[d].sort(key=lambda x: int(x["branch_position"]))
        cidx = self.curve_index.get(rid)
        if cidx is None or cidx >= self.limit_curves:
            for j in range(len(rows)):
                g = gstart + j
                self.row_disp[g] = DISP_UNMATCHED
                self.row_reason[g] = R_NOT_IN_IDX
            return
        n_nodes = int(self.n_nodes[cidx])
        ts = int(self.ts_index[cidx])
        lo, hi = int(self.offsets[cidx]), int(self.offsets[cidx + 1])
        order = np.asarray(self.node_order[lo:hi])
        if len(order) != n_nodes:
            raise BuildError(f"{rid}: node_order slice != n_nodes")
        ts_hits = np.nonzero(order == ts)[0]
        if len(ts_hits) != 1:
            raise BuildError(f"{rid}: ts_index {ts} not a unique node")
        ts_path = int(ts_hits[0])
        a_rows, b_rows = by_dir["A"], by_dir["B"]
        if not a_rows or not b_rows:
            raise BuildError(f"{rid}: matched reaction without both branches")
        g = self.geom_offsets[rid]
        n_atoms = int(g["n_atoms"])
        off = int(g["offset"])
        frames = np.asarray(
            self.coords_mm[off: off + n_nodes * n_atoms]).reshape(
                n_nodes, n_atoms, 3)
        if frames.shape[0] != n_nodes:
            raise BuildError(f"{rid}: geometry frames != n_nodes")

        def propose(kind: str):
            mapping: dict[str, int] = {}
            for r in a_rows:
                p = int(r["branch_position"])
                canon = ts_path + p if kind == "inc" else ts_path - p
                if not 0 <= canon < n_nodes:
                    return None
                mapping[r["frame_key"]] = canon
            for r in b_rows:
                p = int(r["branch_position"])
                if p == 0:
                    continue
                canon = ts_path - p if kind == "inc" else ts_path + p
                if not 0 <= canon < n_nodes:
                    return None
                mapping[r["frame_key"]] = canon
            if len(set(mapping.values())) != len(mapping):
                return None
            return mapping

        valid: dict[str, dict[str, int]] = {}
        for kind in ("inc", "dec"):
            mapping = propose(kind)
            if mapping is None:
                continue
            ok = True
            for r in rows:
                canon = mapping.get(r["frame_key"])
                if canon is None:
                    continue
                node = int(order[canon])
                xtb_xyz = np.asarray(r["coordinates_angstrom"],
                                     dtype=np.float64).reshape(n_atoms, 3)
                diff = float(np.abs(round10(frames[node]) -
                                    round10(xtb_xyz)).max())
                if diff != 0.0:
                    ok = False
                    break
            if ok:
                valid[kind] = mapping
        if len(valid) != 1:
            state["orient_fail"].append({
                "reaction_id": rid, "valid_kinds": sorted(valid),
                "ts_path": ts_path, "n_nodes": n_nodes,
                "a_pmax": int(a_rows[-1]["branch_position"]),
                "b_pmax": int(b_rows[-1]["branch_position"]),
            })
            raise BuildError(f"{rid}: orientation not uniquely resolved by "
                             f"geometry: valid={sorted(valid)}")
        kind = next(iter(valid))
        mapping = valid[kind]
        self.orient[cidx] = ORIENT_INC if kind == "inc" else ORIENT_DEC
        if kind == "inc":
            state["n_inc"] += 1
            if int(a_rows[-1]["branch_length"]) != n_nodes - ts_path or \
                    int(b_rows[-1]["branch_length"]) != ts_path + 1:
                raise BuildError(f"{rid}: branch lengths inconsistent with "
                                 "A_increasing")
        else:
            state["n_dec"] += 1
            if int(a_rows[-1]["branch_length"]) != ts_path + 1 or \
                    int(b_rows[-1]["branch_length"]) != n_nodes - ts_path:
                raise BuildError(f"{rid}: branch lengths inconsistent with "
                                 "A_decreasing")
        self.branch_a_len[cidx] = int(a_rows[-1]["branch_length"])
        self.branch_b_len[cidx] = int(b_rows[-1]["branch_length"])
        self.short_a[cidx] = any(bool(r["short_branch"]) for r in a_rows)
        self.short_b[cidx] = any(bool(r["short_branch"]) for r in b_rows)
        self.row_start_by_curve[cidx] = gstart
        self.row_end_by_curve[cidx] = gend
        for j, r in enumerate(rows):
            gi = gstart + j
            key = r["frame_key"]
            failed = key in self.failures
            p = int(r["branch_position"])
            self.row_curve[gi] = cidx
            if r["direction"] == "B" and p == 0:
                self.row_disp[gi] = DISP_EXCLUDED
                self.row_reason[gi] = R_REPLICA
                state["n_replica"] += 1
                continue
            canon = mapping[key]
            node = int(order[canon])
            xtb_xyz = np.asarray(r["coordinates_angstrom"],
                                 dtype=np.float64).reshape(n_atoms, 3)
            diff = float(np.abs(round10(frames[node]) -
                                round10(xtb_xyz)).max())
            state["n_geom_checked"] += 1
            self.row_gchecked[gi] = 1
            self.row_maxdiff[gi] = diff
            if diff > state["geom_maxdiff"]:
                state["geom_maxdiff"] = diff
            if diff != 0.0:
                state["n_geom_mismatch"] += 1
            self.row_canon[gi] = canon
            if failed:
                self.row_disp[gi] = DISP_EXCLUDED
                self.row_reason[gi] = R_SCF
                state["n_failed_mapped"] += 1
            else:
                self.row_disp[gi] = DISP_JOINED
                self.row_reason[gi] = R_NONE
                state["n_joined"] += 1

    def _structural_checks(self, rid, rows) -> None:
        dirs = {r["direction"] for r in rows}
        if not dirs <= {"A", "B"}:
            raise BuildError(f"{rid}: bad directions {dirs}")
        kinds = {r["branch_start_kind"] for r in rows}
        if not kinds <= {"ts", "near_ts_replica"}:
            raise BuildError(f"{rid}: bad branch_start_kind {kinds}")
        for d in ("A", "B"):
            rs = sorted([r for r in rows if r["direction"] == d],
                        key=lambda x: int(x["branch_position"]))
            if not rs:
                continue
            start = 0 if d == "A" else int(rs[0]["splice_index"]) + 1
            if int(rs[0]["branch_position"]) != 0:
                raise BuildError(f"{rid}/{d}: branch_position does not start at 0")
            if int(rs[0]["frame_index"]) != start:
                raise BuildError(f"{rid}/{d}: frame_index start != branch start")
            for x in rs:
                if int(x["frame_index"]) != start + int(x["branch_position"]):
                    raise BuildError(f"{rid}/{d}: frame_index/branch_position "
                                     "drift")
                if int(x["branch_position"]) >= int(x["branch_length"]):
                    raise BuildError(f"{rid}/{d}: branch_position out of branch")
                if int(x["n_points"]) != len(rs):
                    raise BuildError(f"{rid}/{d}: n_points != sampled rows")
                if int(x["ts_frame_index"]) != 0:
                    raise BuildError(f"{rid}/{d}: unexpected ts_frame_index")
            if int(rs[-1]["branch_position"]) != int(rs[-1]["branch_length"]) - 1:
                raise BuildError(f"{rid}/{d}: branch endpoint not sampled")
        nfr = {int(r["n_frames"]) for r in rows}
        if len(nfr) != 1:
            raise BuildError(f"{rid}: inconsistent n_frames")

    # -- phase 2: frame crosswalk ------------------------------------------
    def _frame_schema(self) -> pa.Schema:
        return pa.schema([
            pa.field("global_frame_id", pa.int64(), nullable=False),
            pa.field("curve_index", pa.int32(), nullable=False),
            pa.field("reaction_id", pa.string(), nullable=False),
            pa.field("curve_id", pa.string(), nullable=False),
            pa.field("canonical_node_index", pa.int32(), nullable=False),
            pa.field("idx_v2_node_index", pa.int32(), nullable=False),
            pa.field("dft_frame_index", pa.int32(), nullable=False),
            pa.field("branch_side", pa.string(), nullable=False),
            pa.field("direction", pa.string(), nullable=False),
            pa.field("xtb_frame_index", pa.int32(), nullable=False),
            pa.field("xtb_frame_key", pa.binary(), nullable=False),
            pa.field("has_xtb", pa.bool_(), nullable=False),
            pa.field("coords_sha256", pa.binary(), nullable=False),
            pa.field("coords_sha256_source", pa.string(), nullable=False),
            pa.field("excluded_reason", pa.string(), nullable=False),
        ])

    def write_crosswalk(self) -> dict:
        log = self.log
        log("[2/5] writing frame_crosswalk.parquet (streaming per curve)")
        schema = self._frame_schema()
        out_path = self.out_dir / "frame_crosswalk.parquet"
        dict_cols = ["reaction_id", "curve_id", "branch_side", "direction",
                     "coords_sha256_source", "excluded_reason"]
        writer = pq.ParquetWriter(out_path, schema, compression="zstd",
                                  use_dictionary=dict_cols,
                                  write_statistics=True)
        buf: dict[str, list] = {f.name: [] for f in schema}
        n_buffered = 0

        def append_chunk() -> None:
            nonlocal n_buffered
            if n_buffered == 0:
                return
            arrays = []
            for f in schema:
                values = buf[f.name]
                if pa.types.is_int64(f.type):
                    arrays.append(pa.array(np.asarray(values, dtype=np.int64),
                                           type=f.type))
                elif pa.types.is_int32(f.type):
                    arrays.append(pa.array(np.asarray(values, dtype=np.int32),
                                           type=f.type))
                elif pa.types.is_boolean(f.type):
                    arrays.append(pa.array(np.asarray(values, dtype=np.bool_),
                                           type=f.type))
                elif pa.types.is_binary(f.type):
                    arrays.append(pa.array(np.asarray(values, dtype="S64"),
                                           type=f.type))
                else:
                    arrays.append(pa.array(np.asarray(values, dtype="U64"),
                                           type=f.type))
            writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
            for k in buf:
                buf[k] = []
            n_buffered = 0

        reaction_reason: dict[str, str] = {}
        for name in ("dft_energy_unavailable", "unsupported_layout",
                     "duplicate_frame_key"):
            for cid in self.xtb_sets[name]:
                reaction_reason[cid] = name

        n_curves = len(self.curve_ids)
        curve_starts = np.zeros(n_curves, dtype=np.int64)
        frame_id = 0
        n_with = n_without = 0
        reason_counts: dict[str, int] = {}
        n_short_frames = 0
        n_hash_checked = n_hash_bad = 0
        hash_bad_examples: list[dict] = []
        t0 = time.perf_counter()

        for cidx, cid in enumerate(self.curve_ids[:self.limit_curves]):
            n_nodes = int(self.n_nodes[cidx])
            curve_starts[cidx] = frame_id
            lo, hi = int(self.offsets[cidx]), int(self.offsets[cidx + 1])
            order = np.asarray(self.node_order[lo:hi])
            ts = int(self.ts_index[cidx])
            ts_hits = np.nonzero(order == ts)[0]
            if len(ts_hits) != 1:
                raise BuildError(f"{cid}: ts not unique in node_order")
            ts_path = int(ts_hits[0])
            orient = int(self.orient[cidx])
            whole_reason = reaction_reason.get(cid, "")
            short_a = bool(self.short_a[cidx])
            short_b = bool(self.short_b[cidx])
            g = self.geom_offsets[cid]
            n_atoms = int(g["n_atoms"])
            off = int(g["offset"])
            frames = np.asarray(
                self.coords_mm[off: off + n_nodes * n_atoms]).reshape(
                    n_nodes, n_atoms, 3)

            keys = [b""] * n_nodes
            has = np.zeros(n_nodes, dtype=bool)
            direction = [""] * n_nodes
            xtb_fi = np.full(n_nodes, -1, dtype=np.int32)
            csha = [b""] * n_nodes
            csha_src = ["idx_v2_geom_recomputed_v1"] * n_nodes
            reason = [whole_reason] * n_nodes
            branch_side = [""] * n_nodes
            dft_fi = np.full(n_nodes, -1, dtype=np.int32)
            assigned = np.zeros(n_nodes, dtype=bool)

            r0 = int(self.row_start_by_curve[cidx])
            r1 = int(self.row_end_by_curve[cidx])
            if r0 >= 0:
                for gi in range(r0, r1):
                    p = int(self.row_p[gi])
                    d = "A" if self.row_dir[gi] == 0 else "B"
                    if d == "B" and p == 0:
                        continue
                    disp = int(self.row_disp[gi])
                    if disp == DISP_UNMATCHED:
                        raise BuildError(f"{cid}: unexpected unmatched row")
                    canon = int(self.row_canon[gi])
                    if assigned[canon]:
                        raise BuildError(f"{cid}: canonical node {canon} "
                                         "claimed twice")
                    assigned[canon] = True
                    if disp == DISP_JOINED:
                        has[canon] = True
                        keys[canon] = bytes(self.row_key[gi])
                        direction[canon] = d
                        xtb_fi[canon] = int(self.row_fi[gi])
                        csha[canon] = bytes(self.row_csha[gi])
                        csha_src[canon] = "xtb_manifest"
                        reason[canon] = ""
                    elif disp == DISP_EXCLUDED and \
                            int(self.row_reason[gi]) == R_SCF:
                        reason[canon] = "scf_not_converged"
                        direction[canon] = d
                        xtb_fi[canon] = int(self.row_fi[gi])
                    else:
                        raise BuildError(f"{cid}: bad row disposition "
                                         f"{disp}/{int(self.row_reason[gi])}")
                if orient == ORIENT_INC:
                    for pos in range(n_nodes):
                        if pos >= ts_path:
                            branch_side[pos] = "A"
                            dft_fi[pos] = pos - ts_path
                        else:
                            branch_side[pos] = "B"
                            dft_fi[pos] = n_nodes - pos
                elif orient == ORIENT_DEC:
                    for pos in range(n_nodes):
                        if pos <= ts_path:
                            branch_side[pos] = "A"
                            dft_fi[pos] = ts_path - pos
                        else:
                            branch_side[pos] = "B"
                            dft_fi[pos] = pos + 1
                if orient in (ORIENT_INC, ORIENT_DEC):
                    for pos in range(n_nodes):
                        if has[pos] or reason[pos]:
                            continue
                        side = branch_side[pos]
                        if (side == "A" and short_a) or (side == "B" and short_b):
                            reason[pos] = "short_branch"
                            n_short_frames += 1
            # xTB frame index cross-check + geometry hashes
            for pos in range(n_nodes):
                node = int(order[pos])
                hb = coords_sha256(frames[node]).encode("ascii")
                if has[pos]:
                    n_hash_checked += 1
                    if hb != csha[pos]:
                        n_hash_bad += 1
                        if len(hash_bad_examples) < 5:
                            hash_bad_examples.append(
                                {"reaction_id": cid, "canonical_node_index": pos,
                                 "xtb_coords_sha256": csha[pos].decode(),
                                 "recomputed": hb.decode()})
                    if orient in (ORIENT_INC, ORIENT_DEC) and \
                            int(xtb_fi[pos]) != int(dft_fi[pos]):
                        raise BuildError(
                            f"{cid}: xTB frame_index {int(xtb_fi[pos])} != "
                            f"derived dft_frame_index {int(dft_fi[pos])} at "
                            f"canonical node {pos}")
                else:
                    csha[pos] = hb
                    csha_src[pos] = "idx_v2_geom_recomputed_v1"

            if n_buffered + n_nodes > CROSSWALK_CHUNK_ROWS:
                append_chunk()
            base = frame_id
            buf["global_frame_id"].extend(range(base, base + n_nodes))
            buf["curve_index"].extend([cidx] * n_nodes)
            buf["reaction_id"].extend([cid] * n_nodes)
            buf["curve_id"].extend([cid] * n_nodes)
            buf["canonical_node_index"].extend(range(n_nodes))
            buf["idx_v2_node_index"].extend(int(v) for v in order)
            buf["dft_frame_index"].extend(int(v) for v in dft_fi)
            buf["branch_side"].extend(branch_side)
            buf["direction"].extend(direction)
            buf["xtb_frame_index"].extend(int(v) for v in xtb_fi)
            buf["xtb_frame_key"].extend(keys)
            buf["has_xtb"].extend(bool(v) for v in has)
            buf["coords_sha256"].extend(csha)
            buf["coords_sha256_source"].extend(csha_src)
            buf["excluded_reason"].extend(reason)
            n_buffered += n_nodes
            if r0 >= 0:
                joined = np.nonzero((self.row_disp[r0:r1] == DISP_JOINED))[0] + r0
                for gi in joined:
                    self.row_global[gi] = base + int(self.row_canon[gi])
            n_has = int(has.sum())
            n_with += n_has
            n_without += n_nodes - n_has
            for rs in reason:
                if rs:
                    reason_counts[rs] = reason_counts.get(rs, 0) + 1
            frame_id += n_nodes
            if (cidx + 1) % 10000 == 0:
                append_chunk()
                log(f"    {cidx + 1}/{n_curves} curves, {frame_id} frames "
                    f"({time.perf_counter() - t0:.0f}s)")
        append_chunk()
        writer.close()
        if frame_id != self.expected_frames:
            raise BuildError(f"wrote {frame_id} frames != expected "
                             f"{self.expected_frames}")
        if n_hash_bad:
            raise BuildError(f"{n_hash_bad} joined frames fail geometry-hash "
                             f"recomputability: {hash_bad_examples[:3]}")
        log(f"    wrote {out_path.name}: {frame_id} rows, "
            f"{out_path.stat().st_size / 1e6:.1f} MB, with_xtb={n_with}, "
            f"no_xtb={n_without} ({time.perf_counter() - t0:.0f}s)")
        self.curve_starts = curve_starts
        self.n_frames_with_xtb = n_with
        self.n_frames_no_xtb = n_without
        self.reason_counts = reason_counts
        self.n_short_frames = n_short_frames
        self.n_hash_checked = n_hash_checked
        self.n_hash_bad = n_hash_bad
        return {}

    # -- phase 3: ledgers --------------------------------------------------
    def write_ledgers(self) -> dict:
        log = self.log
        log("[3/5] writing xtb_row_ledger.parquet + excluded_ledger.parquet")
        N = self.rows
        rxn_per_row = np.empty(N, dtype=object)
        for (gstart, gend), rid in zip(self.group_bounds, self.group_reactions):
            rxn_per_row[gstart:gend] = rid
        dft_fi = np.full(N, -1, dtype=np.int32)
        for cidx in range(len(self.curve_ids)):
            r0 = int(self.row_start_by_curve[cidx])
            r1 = int(self.row_end_by_curve[cidx])
            if r0 < 0:
                continue
            n_nodes = int(self.n_nodes[cidx])
            ts = int(self.ts_index[cidx])
            orient = int(self.orient[cidx])
            joined = np.nonzero(self.row_global[r0:r1] >= 0)[0] + r0
            for gi in joined:
                pos = int(self.row_canon[gi])
                if orient == ORIENT_INC:
                    dft_fi[gi] = pos - ts if pos >= ts else n_nodes - pos
                elif orient == ORIENT_DEC:
                    dft_fi[gi] = ts - pos if pos <= ts else pos + 1
        direction = np.where(self.row_dir == 0, "A", "B").astype("U1")
        branch_start_kind = np.where(self.row_dir == 0, "ts",
                                     "near_ts_replica").astype("U16")
        disp_names = np.array([DISP_NAMES[int(v)] for v in self.row_disp],
                              dtype="U16")
        reason_names = np.array([REASON_NAMES[int(v)] for v in self.row_reason],
                                dtype="U32")
        ledger_schema = pa.schema([
            pa.field("xtb_row_ordinal", pa.int32(), nullable=False),
            pa.field("xtb_frame_key", pa.string(), nullable=False),
            pa.field("reaction_id", pa.string(), nullable=False),
            pa.field("direction", pa.string(), nullable=False),
            pa.field("frame_index", pa.int32(), nullable=False),
            pa.field("branch_position", pa.int32(), nullable=False),
            pa.field("branch_start_kind", pa.string(), nullable=False),
            pa.field("scf_converged", pa.bool_(), nullable=False),
            pa.field("disposition", pa.string(), nullable=False),
            pa.field("reason", pa.string(), nullable=False),
            pa.field("curve_index", pa.int32(), nullable=False),
            pa.field("canonical_node_index", pa.int32(), nullable=False),
            pa.field("dft_frame_index", pa.int32(), nullable=False),
            pa.field("global_frame_id", pa.int64(), nullable=False),
        ])
        ledger_path = self.out_dir / "xtb_row_ledger.parquet"
        writer = pq.ParquetWriter(
            ledger_path, ledger_schema, compression="zstd",
            use_dictionary=["reaction_id", "direction", "branch_start_kind",
                            "disposition", "reason"])
        step = 250_000
        for start in range(0, N, step):
            end = min(start + step, N)
            writer.write_table(pa.Table.from_arrays([
                pa.array(np.arange(start, end, dtype=np.int32)),
                pa.array([bytes(k).decode("ascii")
                          for k in self.row_key[start:end]]),
                pa.array(np.asarray(rxn_per_row[start:end], dtype=object)),
                pa.array(direction[start:end]),
                pa.array(self.row_fi[start:end].astype(np.int32)),
                pa.array(self.row_p[start:end].astype(np.int32)),
                pa.array(branch_start_kind[start:end]),
                pa.array(~self.row_failed[start:end].astype(bool)),
                pa.array(disp_names[start:end]),
                pa.array(reason_names[start:end]),
                pa.array(self.row_curve[start:end].astype(np.int32)),
                pa.array(self.row_canon[start:end].astype(np.int32)),
                pa.array(dft_fi[start:end]),
                pa.array(self.row_global[start:end]),
            ], schema=ledger_schema))
        writer.close()
        log(f"    wrote {ledger_path.name}: {N} rows, "
            f"{ledger_path.stat().st_size / 1e6:.1f} MB")

        # ---- exclusion ledger --------------------------------------------
        rec_type, rec_reason, rec_rid, rec_dir = [], [], [], []
        npt, npa, miss, blen = [], [], [], []
        in_idx, nframes, detail = [], [], []
        for name in ("dft_energy_unavailable", "unsupported_layout",
                     "duplicate_frame_key"):
            for cid in sorted(self.xtb_sets[name]):
                cidx = self.curve_index.get(cid)
                rec_type.append("reaction")
                rec_reason.append(name)
                rec_rid.append(cid)
                rec_dir.append("")
                npt.append(24)
                npa.append(0)
                miss.append(24)
                blen.append(-1)
                in_idx.append(cidx is not None)
                nframes.append(int(self.n_nodes[cidx]) if cidx is not None else 0)
                if name == "dft_energy_unavailable":
                    detail.append(
                        str(self.xtb_reasons.get(cid, "")) or
                        "IRC archive exposes no per-frame energies; no blind "
                        "alignment attempted")
                elif name == "unsupported_layout":
                    detail.append("trajectory layout is not ts_first_dual_branch; "
                                  "not sampled by irc_dense12_v1")
                else:
                    detail.append("idempotency frame_key collision inside the "
                                  "reaction; the whole reaction was excluded")
        # branch-level short_branch records
        branch_records: list[dict] = []
        short_missing_total = 0
        for (gstart, gend), rid in zip(self.group_bounds, self.group_reactions):
            by_dir: dict[str, list[int]] = {"A": [], "B": []}
            for gi in range(gstart, gend):
                by_dir["A" if self.row_dir[gi] == 0 else "B"].append(gi)
            cidx = self.curve_index.get(rid)
            for d, idxs in by_dir.items():
                if not idxs:
                    continue
                if not any(int(self.row_short[gi]) for gi in idxs):
                    continue
                bl = int(self.row_blen[idxs[0]])
                npts = len(idxs)
                gap = 12 - npts
                short_missing_total += gap
                branch_records.append({
                    "reaction_id": rid, "direction": d, "branch_length": bl,
                    "n_points": npts, "missing_points": gap,
                    "cidx": cidx,
                })
                rec_type.append("branch")
                rec_reason.append("short_branch")
                rec_rid.append(rid)
                rec_dir.append(d)
                npt.append(12)
                npa.append(npts)
                miss.append(gap)
                blen.append(bl)
                in_idx.append(cidx is not None)
                nframes.append(max(0, bl - npts) if cidx is not None else 0)
                detail.append("branch shorter than the 18-frame dense12 target; "
                              "sampling truncated to the available frames")
        if short_missing_total != self.xtb_shortfall["short_branch"] and \
                not self.args.smoke_max_shards:
            raise BuildError(f"short-branch gap {short_missing_total} != frozen "
                             f"index shortfall {self.xtb_shortfall['short_branch']}")
        excl_schema = pa.schema([
            pa.field("record_type", pa.string(), nullable=False),
            pa.field("reason", pa.string(), nullable=False),
            pa.field("reaction_id", pa.string(), nullable=False),
            pa.field("direction", pa.string(), nullable=False),
            pa.field("n_points_nominal", pa.int32(), nullable=False),
            pa.field("n_points_actual", pa.int32(), nullable=False),
            pa.field("missing_points", pa.int32(), nullable=False),
            pa.field("branch_length", pa.int32(), nullable=False),
            pa.field("in_idx_v2", pa.bool_(), nullable=False),
            pa.field("n_frames_in_crosswalk", pa.int64(), nullable=False),
            pa.field("detail", pa.string(), nullable=False),
        ])
        excl_path = self.out_dir / "excluded_ledger.parquet"
        pq.write_table(
            pa.Table.from_arrays([
                pa.array(rec_type), pa.array(rec_reason), pa.array(rec_rid),
                pa.array(rec_dir),
                pa.array(np.asarray(npt, dtype=np.int32)),
                pa.array(np.asarray(npa, dtype=np.int32)),
                pa.array(np.asarray(miss, dtype=np.int32)),
                pa.array(np.asarray(blen, dtype=np.int32)),
                pa.array(np.asarray(in_idx, dtype=bool)),
                pa.array(np.asarray(nframes, dtype=np.int64)),
                pa.array(detail),
            ], schema=excl_schema),
            excl_path, compression="zstd",
            use_dictionary=["record_type", "reason", "direction", "detail"])
        log(f"    wrote {excl_path.name}: "
            f"{len(rec_type)} records ({len(branch_records)} short branches), "
            f"{excl_path.stat().st_size / 1e6:.2f} MB")
        self.short_branch_records = len(branch_records)
        self.short_branch_in_idx = sum(1 for r in branch_records
                                       if r["cidx"] is not None)
        self.short_branch_frames = sum(
            max(0, r["branch_length"] - r["n_points"])
            for r in branch_records if r["cidx"] is not None)
        self.excluded_ledger_rows = len(rec_type)
        return {"ledger": ledger_path, "excluded": excl_path}

    # -- phase 4: verify outputs ------------------------------------------
    def verify_outputs(self) -> dict:
        import pyarrow.compute as pc
        log = self.log
        log("[4/5] verifying parquet outputs (row count, ids, uniqueness)")
        out_path = self.out_dir / "frame_crosswalk.parquet"
        pf = pq.ParquetFile(out_path)
        n_rows = pf.metadata.num_rows
        expected = self.expected_frames
        tbl = pq.read_table(out_path, columns=["global_frame_id", "has_xtb"])
        gid = tbl.column("global_frame_id").to_numpy()
        has = tbl.column("has_xtb").to_numpy()
        del tbl
        n_unique = len(np.unique(gid))
        contiguous = bool(n_rows == expected and gid.min() == 0 and
                          gid.max() == expected - 1 and n_unique == expected)
        n_has = int(has.sum())
        del gid, has
        # non-empty xtb_frame_key must equal the has_xtb mask and be unique
        key_tbl = pq.read_table(out_path, columns=["xtb_frame_key"])
        keys = key_tbl.column("xtb_frame_key")
        lens = pc.binary_length(keys).to_numpy()
        n_nonempty = int((lens > 0).sum())
        if n_nonempty != n_has:
            raise BuildError(f"non-empty xtb_frame_key {n_nonempty} != "
                             f"has_xtb {n_has}")
        nonempty = pc.filter(keys, pc.greater(pc.binary_length(keys), 0))
        key_arr = np.asarray(nonempty.to_pylist(), dtype="S64")
        uniq_keys = len(np.unique(key_arr))
        keys_unique = uniq_keys == n_has
        del keys, lens, nonempty, key_arr, key_tbl
        reason_tbl = pq.read_table(out_path, columns=["excluded_reason"])
        vc = pc.value_counts(reason_tbl.column("excluded_reason"))
        bad_reasons = sorted(
            str(v) for v in vc.field("values").to_pylist()
            if str(v) not in FRAME_REASON_VALUES)
        del reason_tbl, vc
        if bad_reasons:
            raise BuildError(f"unexpected excluded_reason values {bad_reasons}")
        n_no = n_rows - n_has
        if n_has != self.n_frames_with_xtb or n_no != self.n_frames_no_xtb:
            raise BuildError("crosswalk has_xtb counts changed on re-read")
        log(f"    rows={n_rows} (expected {expected}), global_frame_id unique & "
            f"contiguous={contiguous}, xtb_frame_key unique={keys_unique}")
        self.verify = {
            "n_rows": n_rows,
            "n_rows_expected": expected,
            "row_count_pass": bool(n_rows == expected),
            "global_frame_id_unique": bool(n_unique == expected),
            "global_frame_id_contiguous_0_to_n_minus_1": contiguous,
            "xtb_frame_key_nonempty_count": n_has,
            "xtb_frame_key_nonempty_unique": bool(keys_unique),
        }
        if not (contiguous and keys_unique and n_rows == expected):
            raise BuildError("output verification failed")
        return self.verify

    # -- phase 5: manifest -------------------------------------------------
    def _file_sha(self, name: str) -> dict:
        p = self.out_dir / name
        return {"path": str(p), "bytes": int(p.stat().st_size),
                "sha256": sha256_file(p)}

    def build_manifest(self, prev_sha: dict, hashes: dict) -> dict:
        self.log("[5/5] writing frame_crosswalk_manifest.json")
        outputs = {
            "frame_crosswalk": self._file_sha("frame_crosswalk.parquet"),
            "xtb_row_ledger": self._file_sha("xtb_row_ledger.parquet"),
            "excluded_ledger": self._file_sha("excluded_ledger.parquet"),
        }
        cur_sha = {
            "frame_crosswalk_parquet_sha256":
                outputs["frame_crosswalk"]["sha256"],
            "xtb_row_ledger_parquet_sha256":
                outputs["xtb_row_ledger"]["sha256"],
            "excluded_ledger_parquet_sha256":
                outputs["excluded_ledger"]["sha256"],
        }
        idem = {
            "previous_run_output_sha256": prev_sha or None,
            "current_run_output_sha256": cur_sha,
            "identical": (None if not prev_sha else prev_sha == cur_sha),
            "note": "parquet+ledger outputs are byte-deterministic; a second "
                    "run over unchanged sources reproduces the same sha256",
        }        # counts
        n_frames = self.expected_frames
        joined = self.n_mapped_joined
        excluded_rows = self.n_replica + self.n_mapped_failed
        unmatched_rows = self.rows - joined - excluded_rows
        total_rows = self.rows
        reconcile_ok = joined + excluded_rows + unmatched_rows == total_rows
        n_curves = self.limit_curves
        n_curves_no_x = int((self.orient[:self.limit_curves] ==
                             ORIENT_UNKNOWN).sum())
        frames_in_no_x = int(self.n_nodes[:self.limit_curves][
            self.orient[:self.limit_curves] == ORIENT_UNKNOWN].sum())
        no_x_curves = [self.curve_ids[i] for i in
                       np.nonzero(self.orient[:self.limit_curves] ==
                                  ORIENT_UNKNOWN)[0]]
        # failure breakdown
        fail_total = self.n_failure_rows
        fail_in_idx = int(((self.row_failed == 1) &
                           (self.row_curve >= 0)).sum())
        fail_mapped = self.n_mapped_failed
        fail_replica = int(((self.row_failed == 1) &
                            (self.row_reason == R_REPLICA)).sum())
        fail_unmatched = fail_total - fail_in_idx
        # reason counts from frame table verification pass (re-read)
        reason_counts = dict(self.reason_counts)
        for name in ("dft_energy_unavailable", "unsupported_layout",
                     "duplicate_frame_key"):
            reason_counts.setdefault(name, 0)
        # shortfall ledger counts
        short_branch_total = self.short_branch_records
        # acceptance
        geom_sample = self._geometry_sample()
        acceptance = {
            "row_count": {
                "value": n_frames,
                "expected": int(self.expected_frames),
                "pass": bool(n_frames == self.expected_frames),
                "note": "= sum(curves.json n_nodes) = len(node_order.npy); "
                        "idx_v2 canonical-frame count"
                        + (" (SMOKE SUBSET)" if self.limit_curves !=
                           len(self.curve_ids) else ""),
            },            "xtb_reconciliation": {
                "xtb_rows_total": total_rows,
                "joined_rows": joined,
                "excluded_rows": excluded_rows,
                "excluded_breakdown": {
                    "near_ts_replica_rows": self.n_replica,
                    "scf_not_converged_mapped_rows": self.n_mapped_failed,
                },
                "unmatched_rows": unmatched_rows,
                "unmatched_breakdown": {
                    "reaction_not_in_idx_v2": unmatched_rows,
                },
                "sum_equals_total": bool(reconcile_ok),
                "pass": bool(reconcile_ok),
            },
            "geometry": {
                "rule": "round both xTB coordinates_angstrom and idx_v2 geom "
                        "to 10 decimals, compare element-wise; equality "
                        "tolerance 0.0 (exact float64 equality after round)",
                "full_numeric_check": {
                    "rows_checked": self.n_geom_checked,
                    "rows_mismatched": self.n_geom_mismatch,
                    "max_abs_diff_after_round10": self.geom_maxdiff,
                },
                "full_hash_recompute_check": {
                    "rows_checked": self.n_hash_checked,
                    "rows_mismatched": self.n_hash_bad,
                    "recipe": "sha256(json.dumps(np.round(coords,10).tolist(),"
                              "separators=(',',':')))",
                },
                "random_sample": geom_sample,
                "pass": bool(self.n_geom_mismatch == 0 and
                             self.geom_maxdiff == 0.0 and
                             self.n_hash_bad == 0 and
                             geom_sample["n_consistent"] ==
                             geom_sample["n_sample"]),
            },
            "uniqueness": {
                **self.verify,
                "pass": bool(self.verify["global_frame_id_unique"] and
                             self.verify["global_frame_id_contiguous_0_to_n_minus_1"]
                             and self.verify["xtb_frame_key_nonempty_unique"]),
            },
            "idempotency": idem,
        }
        join_semantics = {
            "node_order_conclusion": (
                "node_order maps canonical path position -> idx_v2 node index "
                "(the irc_node.node_index / irc_curve node numbering). It is "
                "identity for all 64,195 curves of this delivery because every "
                "curve has endpoint_a_index == 0. It does NOT encode the source "
                "IRC trajectory frame number."),
            "node_order_identity_curves": int(self.n_identity_curves),
            "node_order_nonidentity_curves": int(self.n_nonidentity_curves),
            "xtb_frame_index_is": (
                "source h5 IRC trajectory frame index (TS-first layout: frame 0 "
                "is the TS; n_frames = n_nodes + 1)"),
            "mapping_rule": {
                "A_increasing": {
                    "condition": "ts_index == branch_B_length - 1",
                    "A": "canonical = ts_path + branch_position",
                    "B_p_ge_1": "canonical = ts_path - branch_position",
                    "source_frame": "A side: p; B side: n_nodes - canonical",
                },
                "A_decreasing": {
                    "condition": "ts_index == branch_A_length - 1",
                    "A": "canonical = ts_path - branch_position",
                    "B_p_ge_1": "canonical = ts_path + branch_position",
                    "source_frame": "A side: ts_path - canonical; "
                                    "B side: canonical + 1",
                },
                "B_p_0": "near_ts_replica: duplicates the canonical TS node with "
                         "a different near-TS geometry; no independent canonical "
                         "frame (excluded, ledger reason duplicate_frame_key)",
                "orientation_resolution": "per reaction, by exact round-10 "
                                          "geometry equality of all sampled rows "
                                          "(both candidate orientations tried; "
                                          "the unique fully consistent one wins)",
            },
            "orientation_counts": {
                "A_increasing": int(self.n_orient_inc),
                "A_decreasing": int(self.n_orient_dec),
                "unresolved": 0,
            },
            "source_layout": "ts_first_dual_branch; ts_frame_index=0; "
                             "splice_index = branch_A_length - 1; "
                             "n_frames = n_nodes + 1 (one extra near-TS "
                             "replica frame, excluded)",
            "evidence": {
                "full_geometry_rows_checked": self.n_geom_checked,
                "full_geometry_mismatch": self.n_geom_mismatch,
                "full_hash_rows_checked": self.n_hash_checked,
                "full_hash_mismatch": self.n_hash_bad,
                "derived_dft_frame_index_vs_xtb_frame_index": "compared for "
                    "every joined row; 100% agreement required (build fails "
                    "otherwise)",
            },
            "note": "The V5 plan document assumed node_order could be used to "
                    "reverse-map DFT frame numbers; for this frozen delivery "
                    "that is not true (node_order is identity). The source-frame "
                    "mapping above was established from the geometry pack and "
                    "the xTB rows rather than assumed.",
        }
        coverage = {
            "n_curves": n_curves,
            "n_frames_total": n_frames,
            "n_frames_with_xtb": self.n_frames_with_xtb,
            "n_frames_no_xtb": self.n_frames_no_xtb,
            "n_xtb_rows_total": total_rows,
            "n_xtb_rows_joined": joined,
            "n_xtb_rows_excluded": excluded_rows,
            "n_xtb_rows_unmatched": unmatched_rows,
            "n_curves_with_xtb_rows": int(n_curves - n_curves_no_x),
            "n_curves_without_xtb_rows": n_curves_no_x,
            "n_xtb_reactions_total": int(self.unique_reactions),
            "n_xtb_reactions_matched_to_idx_v2": int(
                n_curves - n_curves_no_x),
            "n_xtb_reactions_unmatched": int(
                self.unique_reactions - (n_curves - n_curves_no_x)),
            "n_frames_in_curves_without_xtb_rows": frames_in_no_x,
            "curves_without_xtb_sample": no_x_curves[:10],
            "n_curves_without_xtb_equals_dev_split": bool(
                self.split_subsets.get("dev") and
                set(no_x_curves) == self.split_subsets["dev"]),
            "n_dev_split_curves": len(self.split_subsets.get("dev") or ()),
            "curves_without_xtb_note": (
                "the 589 curves without xTB rows are exactly the dev split; the "
                "xTB dataset covers train_fit only (reaction-level superset "
                "relationship, no full-pool coverage assumed)"),
            "excluded_reason_counts": reason_counts,
            "short_branch": {
                "n_short_branches_total": short_branch_total,
                "n_short_branches_in_idx_v2": self.short_branch_in_idx,
                "unsampled_frames_marked_in_crosswalk": self.n_short_frames,
                "shortfall_points": self.xtb_shortfall["short_branch"],
                "note": "shortfall_points = sum(12 - n_points) over the 1,671 "
                        "short branches (matches the frozen xTB index)",
            },
            "failures": {
                "n_failure_rows_total": fail_total,
                "n_failure_rows_in_idx_v2": fail_in_idx,
                "n_failure_rows_mapped_to_canonical_frame": fail_mapped,
                "n_failure_rows_near_ts_replica": fail_replica,
                "n_failure_rows_unmatched": fail_unmatched,
                "frames_marked_scf_not_converged": reason_counts.get(
                    "scf_not_converged", 0),
            },
            "shortfall_reconciliation": {
                "nominal_points": self.xtb_nominal,
                "actual_manifest_rows": self.xtb_actual,
                "gap_points": self.xtb_nominal - self.xtb_actual,
                "buckets": self.xtb_shortfall,
            },
            "direction_counts": {
                "A_rows": int((self.row_dir == 0).sum()),
                "B_rows": int((self.row_dir == 1).sum()),
            },
        }
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "crosswalk_schema": CROSSWALK_SCHEMA,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "generated_at_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "build": {
                "script_path": str(Path(__file__).resolve()),
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "command": self.args.command_line,
                "elapsed_s": round(time.perf_counter() - self.t0, 1),
                "stage": self.args.stage,
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pyarrow": pa.__version__,
                "log_path": self.log_path,
                "warnings": self.warnings,
            },
            "sources": hashes,
            "input_counts": {
                "idx_v2": {
                    "n_curves": n_curves,
                    "n_frames": n_frames,
                    "n_intents": int(self.binding.get("n_intents", -1)),
                    "n_samples": int(self.binding.get("n_samples", -1)),
                },
                "xTB": {
                    "n_manifest_rows": int(self.xtb_index.get("n_manifest_rows")),
                    "n_reactions": int(self.xtb_index.get("n_reactions")),
                    "n_reactions_read": int(self.xtb_index.get("n_reactions_read")),
                    "n_branches": int(self.xtb_index.get("n_branches")),
                    "n_success": int((self.xtb_audit or {}).get(
                        "counts", {}).get("n_success", -1)),
                    "n_failure": fail_total,
                    "n_short_branches": int(self.xtb_index.get(
                        "n_short_branches")),
                    "n_manifest_shards": len(self.counts_per_shard),
                },
            },
            "join_semantics": join_semantics,
            "coverage": coverage,
            "exclusions": {
                "reaction_level": {
                    name: {
                        "n_reactions": len(self.xtb_sets[name]),
                        "n_reactions_in_idx_v2": len(
                            self.xtb_sets[name] & set(self.curve_index)),
                        "n_frames_in_crosswalk": int(sum(
                            int(self.n_nodes[i]) for i, cid in
                            enumerate(self.curve_ids)
                            if cid in self.xtb_sets[name])),
                        "points_nominal": len(self.xtb_sets[name]) * 24,
                        "shortfall_points_frozen": self.xtb_shortfall.get(name, 0),
                    }
                    for name in ("dft_energy_unavailable", "unsupported_layout",
                                 "duplicate_frame_key")
                },
                "short_branch": coverage["short_branch"],
                "failures": coverage["failures"],
                "ledger_path": str(self.out_dir / "excluded_ledger.parquet"),
                "note": "the 688 dft_energy_unavailable / 452 unsupported_layout "
                        "/ 1 duplicate_frame_key reactions are all outside this "
                        "idx_v2 delivery (intersection 0), so they contribute no "
                        "canonical frame rows; they are recorded explicitly in "
                        "excluded_ledger.parquet and counted here. The 1,671 "
                        "short branches inside idx_v2 are recorded per branch "
                        "and their unsampled frames are marked "
                        "excluded_reason=short_branch where determinable.",
            },
            "acceptance": acceptance,
            "outputs": outputs,
            "output_schema": {
                "frame_crosswalk": {
                    "schema": CROSSWALK_SCHEMA,
                    "rows": n_frames,
                    "columns": {f: str(t) for f, t in
                                zip([f.name for f in self._frame_schema()],
                                    [f.type for f in self._frame_schema()])},
                    "global_frame_id": "int64, 0..N_frames-1, unique contiguous; "
                                       "order = curves.json curve order then "
                                       "canonical node_order dense expansion",
                    "dft_frame_index": "source IRC trajectory frame index "
                                       "(same numbering as xTB frame_index); "
                                       "-1 for the 589 dev curves without xTB "
                                       "rows where orientation is unknowable",
                    "branch_side": "A/B derived from the resolved orientation; "
                                   "empty when orientation is unknown",
                    "direction": "xTB row direction (A/B) for joined frames and "
                                 "for frames whose xTB single point failed "
                                 "(scf_not_converged); empty for frames with no "
                                 "xTB row",
                    "coords_sha256": "32-byte binary; xTB coords_sha256 when "
                                     "joined, else recomputed from idx_v2 geom "
                                     "with the same recipe (see "
                                     "coords_sha256_source)",
                    "xtb_frame_key": "32-byte binary sha256 hex (ASCII); empty "
                                     "when no usable xTB label",
                },
                "xtb_row_ledger": {
                    "schema": LEDGER_SCHEMA,
                    "rows": total_rows,
                    "note": "every xTB manifest row with disposition and reason; "
                            "unmatched rows carry reason=reaction_not_in_idx_v2 "
                            "and negative frame ids",
                },
                "excluded_ledger": {
                    "schema": EXCLUDED_SCHEMA,
                    "rows": self.excluded_ledger_rows,
                    "note": "reaction-level + branch-level exclusions with point "
                            "accounting",
                },
            },
            "failure_state": None,
            "boundaries": [
                "Read-only on every source; this build deletes/moves nothing.",
                "xTB covers train_fit only; the 589 dev curves have no xTB rows. "
                "No full-pool coverage is assumed.",
                "The 2,877,830 xTB rows are accounted row-by-row in "
                "xtb_row_ledger.parquet; unmatched rows are reactions that are "
                "not part of this idx_v2 delivery.",
                "The near_ts_replica rows duplicate the canonical TS node with a "
                "different near-TS geometry; they are excluded from the join "
                "(reason duplicate_frame_key) and never overwrite the exact TS "
                "label.",
                "No energy/gradient values are copied into the crosswalk; it is a "
                "pure frame-level key map (labels stay in the xTB package).",
                "Do not use the xTB dataset for online inference, production "
                "decisions or absolute cross-method energy subtraction "
                "(offline_only=true / truth_assisted=true).",
            ],
        }
        return manifest

    def _geometry_sample(self) -> dict:
        rng = np.random.default_rng(int(self.args.seed))
        candidates = np.nonzero(self.row_gchecked == 1)[0]
        n_sample = min(int(self.args.n_verify_sample), len(candidates))
        picked = rng.choice(candidates, size=n_sample, replace=False)
        diffs = np.asarray([float(self.row_maxdiff[g]) for g in picked])
        records = []
        for g in picked[:20]:
            cidx = int(self.row_curve[g])
            canon = int(self.row_canon[g])
            records.append({
                "xtb_row_ordinal": int(g),
                "reaction_id": self.curve_ids[cidx] if cidx >= 0 else None,
                "curve_index": cidx,
                "canonical_node_index": canon,
                "dft_frame_index": (int(self.row_fi[g])
                                    if self.row_fi is not None else None),
                "direction": "A" if self.row_dir[g] == 0 else "B",
                "branch_position": int(self.row_p[g]),
                "max_abs_diff_round10": float(self.row_maxdiff[g]),
                "xtb_frame_key": bytes(self.row_key[g]).decode("ascii"),
            })
        return {
            "seed": int(self.args.seed),
            "n_sample": int(n_sample),
            "n_consistent": int((diffs == 0.0).sum()),
            "consistent_rate": (float((diffs == 0.0).mean())
                                if n_sample else None),
            "max_abs_diff_round10": float(diffs.max()) if n_sample else None,
            "tolerance": 0.0,
            "rows_available": int(len(candidates)),
            "example_rows": records,
        }

    # -- orchestration -----------------------------------------------------
    def run(self) -> int:
        self.log(f"build_frame_crosswalk start; cwd={Path.cwd()}")
        out_names = ["frame_crosswalk.parquet", "xtb_row_ledger.parquet",
                     "excluded_ledger.parquet"]
        prev_sha = {
            name.removesuffix(".parquet") + "_parquet_sha256": sha_or_none(
                self.out_dir / name)
            for name in out_names
        }
        if any(v is None for v in prev_sha.values()):
            prev_sha = None
        hashes = None
        self.load_sources()
        hashes = self.collect_hashes()
        self.index_xtb_rows()
        if self.args.stage == "index":
            self.log("stage=index done (no outputs written)")
            return 0
        self.write_crosswalk()
        self.write_ledgers()
        self.verify_outputs()
        manifest = self.build_manifest(prev_sha, hashes)
        manifest_path = self.out_dir / "frame_crosswalk_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        self.log(f"wrote {manifest_path}")
        summary = {
            "rows": manifest["acceptance"]["row_count"]["value"],
            "n_frames_with_xtb": self.n_frames_with_xtb,
            "n_frames_no_xtb": self.n_frames_no_xtb,
            "xtb_rows_joined": manifest["acceptance"]["xtb_reconciliation"][
                "joined_rows"],
            "xtb_rows_excluded": manifest["acceptance"]["xtb_reconciliation"][
                "excluded_rows"],
            "xtb_rows_unmatched": manifest["acceptance"]["xtb_reconciliation"][
                "unmatched_rows"],
            "geometry_consistent_rate": manifest["acceptance"]["geometry"][
                "random_sample"]["consistent_rate"],
            "unique_ok": manifest["acceptance"]["uniqueness"]["pass"],
            "idempotent": manifest["acceptance"]["idempotency"]["identical"],
            "outputs": manifest["outputs"],
        }
        print(json.dumps(summary, indent=2), flush=True)
        return 0


def write_fail_manifest(build: CrosswalkBuild, exc: BaseException) -> None:
    """Best-effort FAIL manifest so a failed run is never silent."""
    try:
        path = build.out_dir / "frame_crosswalk_manifest.json"
        doc = {
            "schema": MANIFEST_SCHEMA,
            "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                              time.gmtime()),
            "failure_state": {
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "log_path": build.log_path,
            },
            "build": {
                "script_path": str(Path(__file__).resolve()),
                "script_sha256": sha256_file(Path(__file__).resolve()),
                "command": build.args.command_line,
                "elapsed_s": round(time.perf_counter() - build.t0, 1),
            },
            "acceptance": {"pass": False},
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
    except Exception:  # pragma: no cover - best effort only
        pass


def parse_args(argv: list[str]) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--idx-root", default="/pes2ts_data/v53mp_50k_data")
    p.add_argument("--xtb-root",
                   default=str(repo_root / "data/raw/"
                               "pes2ts_xtb_singlepoints_train_fit_v1"))
    p.add_argument("--out-dir", default="/pes2ts_datastore/01_master_index")
    p.add_argument("--stage", choices=("all", "index"), default="all",
                   help="'index' runs the xTB index + join validation only")
    p.add_argument("--seed", type=int, default=20261123)
    p.add_argument("--n-verify-sample", type=int, default=1000)
    p.add_argument("--log", default=None,
                   help="log file path (default: none; WSL runner redirects)")
    p.add_argument("--smoke-max-shards", type=int, default=0,
                   help=argparse.SUPPRESS)
    p.add_argument("--smoke-max-curves", type=int, default=0,
                   help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    args.command_line = " ".join([sys.executable, *sys.argv])
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    build = CrosswalkBuild(args)
    try:
        rc = build.run()
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        build.log(f"FAIL: {type(exc).__name__}: {exc}")
        write_fail_manifest(build, exc)
        rc = 1
    finally:
        build.close()
    return rc


if __name__ == "__main__":
    sys.exit(main())
