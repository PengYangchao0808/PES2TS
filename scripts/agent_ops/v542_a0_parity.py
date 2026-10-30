#!/usr/bin/env python3
"""V5.4.2 A0 parity certificate CLI (WP1 lane A; ADR-0033 §2.2).

Strict-audit load + CPU bitwise forward parity of ``V542Model`` (A0 =
``A0LegacyBackbone`` + shared ``DirectMoveDecoder`` +
``ReactionProcessHeadV53MP``) against the legacy reference
(``StarEGNNV3`` + ``ReactionProcessHeadV53MP`` = ``KeyAtomV53MPModel``
direct forward) on the SIX pinned checkpoints of config
``backbone_contract.a0_migration.parity.assets`` (3 V541 E0 S17/S29/S43
``runs/`` 37,900-update finals = primary evidence + 3 V53MP C S17/S29/S43
``outputs/pesn/v53mp/`` S3 formal 20k-tier 15,400-update auxiliary).

Certificate scope — READ THIS FIRST
===================================
This certificate proves **forward + evaluator-migration parity ONLY**: the
A0 adapter + unified decoder reproduce the legacy five outputs
(``u, z, q, endpoint_logit, order_score``) and ``forward_move`` BIT-for-bit
on real inner-val batches, AND the frozen evaluator ``evaluate_v53mp``
(H2/H3 inner-val metrics) recomputed through the V542 wrapper equals the
same evaluator run on the legacy reference bitwise or within 1e-9
absolute — with a clean strict-load audit.  It is **NOT**
evidence of E0 training-comparability reuse — new-interface construction
RNG, shared-``w_v`` ownership and id-based param-groups change the
37,900-update training trajectory, which is exactly why ADR-0033 §2.2
freezes ``decision=retrain_a0_x3`` (V541 E0 ×3 stays parity/historical
reference ONLY).  Do not cite this report as a trainability-reuse result.

Asset discipline (fix-5 C3)
===========================
All six assets are REQUIRED.  Every asset's path must be one of the pinned
entries and its SHA-256 must equal the pinned value — a missing file, an
unpinned path, a SHA mismatch or a smoke/non-pinned checkpoint is a TYPED
failure (exit 1).  Nothing is ever silently skipped.

Certificate completeness (gate-1 C2)
====================================
A report is a CERTIFICATE only when ALL SIX required assets ran with no
``--asset`` filter and no ``--skip-gpu``.  The report carries explicit
``certificate_complete`` / ``only`` / ``skip_gpu`` / ``gpu_disposition``
fields; a certificate is PASS only when all 6 assets passed, CPU bitwise
held, the evaluator leg passed and the GPU disposition is ``pass`` or
``inconclusive``.  ``--asset`` / ``--skip-gpu`` runs are DIAGNOSTIC: the
CLI prints ``DIAGNOSTIC (not a certificate)`` and exits non-zero — a
partial run can never be mistaken for a six-asset certificate.

Strict-load audit in the certificate (gate-1 C1)
================================================
Per asset the report persists the FULL ``load_a0_strict`` audit record:
``source_path`` / ``source_sha256`` (the actual verified digest),
``mapping_table_sha256`` (the CANONICAL digest — gate-0 advisory A3
closed: same frozen recipe as every V5.4.2 identity field, see
``pes2ts_core.pesn.models.v542.identity.canonical_sha256``),
``n_source_keys`` (43) / ``n_mapped_targets`` (45), ``dual_write_aliases``,
``target_key_set``, ``torch_equal_verified`` (count) and empty
``missing``/``unexpected``/``shape_mismatch``/``dtype_mismatch``.  The
report schema is ``pes2ts_v542_a0_parity_report_v1`` — an EVIDENCE
superset of the certificate template ``pes2ts_v542_parity_certificate_v1``
(see ``report_schema_note`` for the exact field mapping; no silent
divergence).

Real-batch coverage (gate-1 C3)
===============================
Inner-val batches are selected DETERMINISTICALLY so the required
categories ``start`` / ``pre_TS`` / ``post_TS`` / ``endpoint`` / ``forward``
/ ``reverse`` / ``small_graph`` (<= 8 atoms) / ``large_graph`` (>= 24
atoms) are ACTUALLY covered per asset (a missing required category is a
TYPED ``V542_PARITY_COVERAGE`` failure).  ``isolated_atom`` and
``zero_in_degree_node`` are MEASURED per batch and recorded as counts +
booleans (the 50K pool's compact molecules may contain zero such nodes —
the measured count is the evidence, not an a-priori claim).

Evaluator no-op guard (gate-1 A7)
=================================
The evaluator leg requires every ``EVALUATOR_TOP_KEYS`` top-level key on
both sides AND ``n_float_leaves > 0`` — a future no-op evaluator cannot
pass by comparing empty dicts.

Parity discipline (ADR-0033 §2.2)
=================================
* data view: the frozen V542 50K pool — the EXISTING V541/V53MP loader
  (``ColumnarKeyAtomDatasetV3`` + ``v53mp_pool_split`` +
  ``_batch_from_positions`` + ``forward_view_v3``); no new feature
  pipeline is built here;
* batches: >= 8 REAL inner-val batches per run (frozen minimum), built
  from the seed's inner-val slice;
* CPU: ``torch.set_num_threads(1)``, eval mode, fp32 — PASS is decided by
  ``torch.equal`` bitwise identity on all five outputs AND on
  ``forward_move`` {u, z}; ``q is order_score`` is re-asserted;
* GPU (skipped cleanly when CUDA is unavailable): hard caps
  ``atol=1e-6``/``rtol=1e-5`` with TF32 DISABLED — never loosened.  The
  run-to-run floor (same model, same batch, two consecutive GPU runs) is
  recorded as a DIAGNOSTIC: if the floor already exceeds the cap, GPU
  parity is reported ``inconclusive`` (not a failure) and the CPU bitwise
  result remains the decisive evidence;
* evaluator (config ``parity.evaluator_parity =
  recompute_H2_H3_vs_V541_archive_bitwise_or_le_1e-9``): the FROZEN
  evaluator ``key_atom_v53mp_train_v3.evaluate_v53mp`` is run over the
  SAME inner-val positions twice — once on the strict-audit-loaded
  ``V542Model`` wrapper, once on the legacy reference from the same
  checkpoint — with identical label tables
  (``ColumnarKeyAtomDatasetV3.label_tables()``, the same lazy v52 +
  endpoint tables the campaign driver hands the frozen trainer), the same
  ``_FeatureLRU``/``_CacheView`` feature cache, the same
  ``V53MPTrainerConfig`` (device=cpu), the same ``train_q_mean`` and
  ``tau_end=None`` (no tau calibration is invented here).  EVERY leaf of
  the resulting ``evaluate_v53mp`` output is compared: bitwise-equal or
  |delta| <= 1e-9 absolute passes; anything else is a HARD fail printing
  key + values + max abs delta.  Counts/masks/strings/bools/``None``
  compare EXACTLY (never within tolerance).  Compared top-level keys:
  ``schema``, ``move`` (incl. ``key_rmse_over_zero``, ``sre_and_sign``,
  ``baselines``, ``guards``, ``pair_ownership_audit``),
  ``move_guard_vs_r3``, ``q_head``, ``ordering``, ``endpoint_head``,
  ``run_outcome``, ``n_dev_samples``, ``n_dev_curves`` (headline
  certificate metrics map: rmse_zero → ``move.key_rmse_over_zero
  .median_rmse_over_zero``; sign → ``move.sre_and_sign
  .sign_accuracy_cluster_median``; dead_fraction → ``move.sre_and_sign
  .dead_zone_fraction``; endpoint_auprc → ``endpoint_head.auprc``;
  q_corr → ``q_head.corr_curve_median``; the template key
  ``fixed_coverage_direction`` belongs to the V541 dev evaluator and has
  no leaf in the frozen V5.3-MP ``evaluate_v53mp`` — the frozen
  evaluator's direction metric IS the H2 sign accuracy above).

Evaluator-leg coverage (per asset class)
========================================
All six pinned assets are covered: both source tiers (V541 E0 finals and
V53MP C finals) load into the identical ``KeyAtomV53MPModel`` /
``V542Model`` pair and share one frozen evaluator — there is no
asset-class-specific dependency.  The evaluator leg's ONLY hard
dependency is the frozen 50K data view (tables_dir + split manifest +
label columns) — the same dependency the batch legs already have.  When
that dependency is absent the evaluator leg raises the TYPED failure
``EVALUATOR_PARITY_UNAVAILABLE`` (CLI: typed entry failure) — never a
silent skip; when the artifacts ARE present any metric mismatch is a
hard fail.

Exit codes: 0 = CERTIFICATE PASS (all six required assets passed, GPU
disposition pass/inconclusive); 1 = certificate FAIL or DIAGNOSTIC run
(``--asset``/``--skip-gpu`` runs always exit 1 with a
``DIAGNOSTIC (not a certificate)`` overall line — they never certify);
2 = refused (config unreadable / no pinned assets / usage error).

CLI::

    python scripts/agent_ops/v542_a0_parity.py \\
        [--config config/campaigns/pes2ts_v542_backbone_50k.yaml] \\
        [--asset ID ...] [--skip-gpu] [--out NEW_JSON_PATH]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pes2ts_core.pesn.contracts.key_atom_v3 import (  # noqa: E402
    EDGE_INDEX_KEY,
    NODE_POS_KEY,
    forward_view_v3,
)
from pes2ts_core.pesn.models.key_atom_heads_v3 import (  # noqa: E402
    KeyAtomV53MPModel,
)
from pes2ts_core.pesn.models.v542 import (  # noqa: E402
    FORWARD_KEYS_V542,
    V542Model,
    load_a0_strict,
)

#: Frozen GPU tolerance caps (config thresholds.gpu_parity_*; NEVER loosened).
GPU_ATOL: float = 1.0e-6
GPU_RTOL: float = 1.0e-5

#: Frozen evaluator-parity tolerance (config
#: ``backbone_contract.a0_migration.parity.evaluator_parity =
#: recompute_H2_H3_vs_V541_archive_bitwise_or_le_1e-9``; NEVER loosened).
EVALUATOR_ATOL: float = 1.0e-9

#: Documented comparison surface of the evaluator leg: every leaf of the
#: ``evaluate_v53mp`` output is compared (bitwise or <= ``EVALUATOR_ATOL``
#: absolute for floats; exact for ints/bools/strings/None).  These are the
#: top-level keys walked by ``compare_evaluator_metrics``.
EVALUATOR_TOP_KEYS: tuple = (
    "schema",
    "move",              # key_rmse_over_zero, sre_and_sign, baselines, ...
    "move_guard_vs_r3",
    "q_head",
    "ordering",
    "endpoint_head",
    "run_outcome",
    "n_dev_samples",
    "n_dev_curves",
)

#: Frozen minimum of REAL inner-val batches per asset (ADR-0033 §2.2).
MIN_INNER_VAL_BATCHES: int = 8
#: Positions (frames) per parity batch — matches batch size class [8, 4].
BATCH_POSITIONS: int = 8

#: Required REAL-BATCH coverage categories (gate-1 C3).  The harness
#: deterministically SELECTS inner-val positions so every category below
#: is covered (first hit in sorted inner-val order per missing category,
#: appended to the frozen first-8-chunk base); a missing required
#: category is a typed ``V542_PARITY_COVERAGE`` failure.
REQUIRED_COVERAGE_CATEGORIES: tuple = (
    "start", "pre_TS", "post_TS", "endpoint",
    "forward", "reverse", "small_graph", "large_graph")
#: Coverage categories that are MEASURED (never claimed a priori): the
#: counts are recorded per asset; zero is the expected measurement on
#: the fully-connected 50K molecules.
MEASURED_ONLY_COVERAGE_CATEGORIES: tuple = ("isolated_atom",
                                            "zero_in_degree_node")
#: small_graph := n_atoms <= this; large_graph := n_atoms >= this.
SMALL_GRAPH_MAX_ATOMS: int = 8
LARGE_GRAPH_MIN_ATOMS: int = 24

#: Report schema id (new V542 audit artifact; nothing historical).
REPORT_SCHEMA: str = "pes2ts_v542_a0_parity_report_v1"

DEFAULT_CONFIG: str = "config/campaigns/pes2ts_v542_backbone_50k.yaml"

EXIT_PASS, EXIT_FAIL, EXIT_REFUSED = 0, 1, 2


class V542ParityError(RuntimeError):
    """Typed parity-harness refusal (``code`` + ``detail``)."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = str(code)
        self.detail = str(detail)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_of(asset_id: str) -> int:
    """Seed from a pinned asset id of the form ``..._S17``."""
    token = str(asset_id).rsplit("_S", 1)
    if len(token) != 2 or not token[1].isdigit():
        raise V542ParityError(
            "V542_PARITY_ASSET_ID",
            f"asset id {asset_id!r} does not end with _S<seed>")
    return int(token[1])


def load_pinned_assets(config_path: Path) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Read the frozen parity block; NOTHING may be unpinned.

    Returns ``(config, assets)``.  Typed refusals: ``V542_PARITY_CONFIG``
    (unreadable config / missing parity block), ``V542_PARITY_ASSETS``
    (not exactly the six required pinned assets).
    """
    import yaml

    if not config_path.is_file():
        raise V542ParityError(
            "V542_PARITY_CONFIG", f"config not found: {config_path}")
    try:
        cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - typed conversion
        raise V542ParityError(
            "V542_PARITY_CONFIG", f"config unreadable: {exc}") from exc
    if not isinstance(cfg, Mapping):
        raise V542ParityError("V542_PARITY_CONFIG", "config is not a mapping")
    parity = (((cfg.get("backbone_contract") or {}).get("a0_migration")
               or {}).get("parity")) or {}
    assets = parity.get("assets")
    if not isinstance(assets, list) or len(assets) != 6:
        raise V542ParityError(
            "V542_PARITY_ASSETS",
            f"parity.assets must list exactly the 6 pinned checkpoints, "
            f"got {len(assets) if isinstance(assets, list) else assets!r}")
    normalized: List[Dict[str, Any]] = []
    for raw in assets:
        if not isinstance(raw, Mapping):
            raise V542ParityError(
                "V542_PARITY_ASSETS", f"asset entry is not a mapping: {raw!r}")
        for field in ("id", "path", "sha256", "required", "source_tier",
                      "training_updates"):
            if field not in raw:
                raise V542ParityError(
                    "V542_PARITY_ASSETS",
                    f"asset {raw.get('id')!r} misses pinned field {field!r}")
        if raw["required"] is not True:
            raise V542ParityError(
                "V542_PARITY_ASSETS",
                f"asset {raw['id']!r} must be required=true (all six are)")
        sha = str(raw["sha256"])
        if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
            raise V542ParityError(
                "V542_PARITY_ASSETS",
                f"asset {raw['id']!r} sha256 {sha!r} is not a pinned hex digest")
        path = str(raw["path"])
        if "smoke" in path.split("/"):
            raise V542ParityError(
                "V542_PARITY_ASSETS",
                f"asset {raw['id']!r} points at a smoke path ({path}); "
                "smoke and non-pinned checkpoints are forbidden")
        normalized.append(dict(raw))
    ids = [a["id"] for a in normalized]
    if len(set(ids)) != len(ids):
        raise V542ParityError(
            "V542_PARITY_ASSETS", f"duplicate asset ids: {ids}")
    return dict(cfg), normalized


def verify_assets(assets: Sequence[Mapping[str, Any]], *,
                  root: Path = PROJECT_ROOT) -> Dict[str, Path]:
    """path-exists + SHA-256 match for every pinned asset (fail-closed)."""
    resolved: Dict[str, Path] = {}
    for asset in assets:
        rel = str(asset["path"])
        path = Path(rel) if Path(rel).is_absolute() else root / rel
        if not path.is_file():
            raise V542ParityError(
                "V542_PARITY_ASSET_MISSING",
                f"asset {asset['id']}: pinned checkpoint not found at {path}")
        actual = sha256_file(path)
        if actual != str(asset["sha256"]):
            raise V542ParityError(
                "V542_PARITY_SHA_MISMATCH",
                f"asset {asset['id']}: sha256 {actual} != pinned "
                f"{asset['sha256']} for {path}")
        resolved[str(asset["id"])] = path
    return resolved


# ---------------------------------------------------------------------------
# frozen data view: the EXISTING V541/V53MP loader (no new pipeline)
# ---------------------------------------------------------------------------
_DATASET_CACHE: Dict[str, Any] = {}
_BATCH_CACHE: Dict[tuple, List[Dict[str, Any]]] = {}


def _open_frozen_dataset(cfg: Mapping[str, Any]) -> Any:
    """Open the frozen 50K columnar pool READ-ONLY (cached per process)."""
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import LABEL_STATUSES
    from pes2ts_core.pesn.offline.data.key_atom_dataset_v3_col import (
        ColumnarKeyAtomDatasetV3,
    )

    tables_dir = str((cfg.get("data_contract") or {}).get("tables_dir")
                     or "/pes2ts_data/v53mp_50k_data")
    split = str((cfg.get("split_contract") or {}).get("split_manifest")
                or "/pes2ts_data/v53mp_50k_data/split_manifest_v53mp.json")
    for path in (Path(tables_dir) / "data_manifest.json",
                 Path(tables_dir) / "teacher_manifest.json", Path(split)):
        if not path.is_file():
            raise V542ParityError(
                "V542_PARITY_DATA",
                f"frozen data view artifact missing: {path}")
    if tables_dir not in _DATASET_CACHE:
        _DATASET_CACHE[tables_dir] = ColumnarKeyAtomDatasetV3(
            tables_dir, split_manifest=str(split),
            label_statuses=LABEL_STATUSES, cache_curves=65000)
    return _DATASET_CACHE[tables_dir]


def _coverage_categories_of_record(dataset: Any, rec: Any) -> set:
    """Cheap record-level coverage categories for ONE inner-val position.

    Signals come straight from the existing data view (no new pipeline):
    frame role from ``node_index`` vs the trusted ``ts_index``/curve
    endpoints; direction from ``intent.direction``; graph size from
    ``len(atomic_numbers)``.
    """
    cid = str(rec.curve_id)
    ni = int(rec.node_index)
    ts = int(dataset.curve_ts_index(cid))
    a_idx, b_idx = (int(x) for x in dataset.curve_endpoints(cid))
    n_atoms = len(rec.atomic_numbers)
    cats = set()
    if ni == a_idx:
        cats.add("start")
    if ni == b_idx:
        cats.add("endpoint")
    if ni < ts:
        cats.add("pre_TS")
    if ni > ts:
        cats.add("post_TS")
    direction = str(getattr(rec.intent, "direction", ""))
    if direction == "forward":
        cats.add("forward")
    elif direction == "reverse":
        cats.add("reverse")
    if n_atoms <= SMALL_GRAPH_MAX_ATOMS:
        cats.add("small_graph")
    if n_atoms >= LARGE_GRAPH_MIN_ATOMS:
        cats.add("large_graph")
    return cats


def _graph_level_coverage_counts(batches: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """MEASURED (never claimed) graph-structural counts over the batches.

    ``isolated_atom``: a node row that appears in NEITHER edge_index row;
    ``zero_in_degree_node``: a node row that never appears as an edge
    ``dst``.  The v3 edge set is bidirected (every pair contributes both
    directions), so zero-in-degree implies isolated here — both counts
    are recorded anyway as the template asks for both signals.
    """
    n_isolated = 0
    n_zero_in = 0
    n_nodes_total = 0
    for batch in batches:
        pos = batch[NODE_POS_KEY]
        edge_index = batch[EDGE_INDEX_KEY]
        n = int(pos.shape[0])
        n_nodes_total += n
        if int(edge_index.shape[1]) == 0:
            n_isolated += n
            n_zero_in += n
            continue
        involved = torch.unique(edge_index.reshape(-1))
        n_isolated += n - int(involved.numel())
        dst = edge_index[1]
        n_zero_in += n - int(torch.unique(dst).numel())
    return {
        "isolated_atom_count": int(n_isolated),
        "zero_in_degree_node_count": int(n_zero_in),
        "n_node_rows": int(n_nodes_total),
        "isolated_atom_present": bool(n_isolated > 0),
        "zero_in_degree_node_present": bool(n_zero_in > 0),
    }


def inner_val_selection(cfg: Mapping[str, Any], seed: int, *,
                        n_batches: int = MIN_INNER_VAL_BATCHES,
                        positions_per_batch: int = BATCH_POSITIONS,
                        ) -> Dict[str, Any]:
    """Deterministic coverage-aware inner-val selection (cached).

    Returns ``{"positions", "batches", "coverage"}``:

    * ``positions`` — the FROZEN base (first ``n_batches *
      positions_per_batch`` sorted inner-val positions, exactly as the
      pre-gate-1 harness selected them) PLUS one appended position per
      required coverage category missing from the base (first hit in
      sorted inner-val order — deterministic by construction, no RNG);
    * ``batches`` — the base batches (unchanged chunking) plus one
      appended coverage batch holding the extra positions;
    * ``coverage`` — per-category position counts + presence booleans
      over the SELECTED positions, plus the measured graph-structural
      counts (``isolated_atom`` / ``zero_in_degree_node``) over the
      built batches.

    Typed failure ``V542_PARITY_COVERAGE`` when a REQUIRED category
    (gate-1 C3) cannot be found anywhere in the inner-val slice.
    """
    if n_batches < MIN_INNER_VAL_BATCHES:
        raise V542ParityError(
            "V542_PARITY_BATCHES",
            f"at least {MIN_INNER_VAL_BATCHES} real inner-val batches are "
            f"frozen (got n_batches={n_batches})")
    key = (str(cfg.get("campaign_id")), int(seed), int(n_batches),
           int(positions_per_batch))
    if key in _BATCH_CACHE:
        return _BATCH_CACHE[key]

    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        _FeatureLRU,
        _batch_from_positions,
        v53mp_pool_split,
    )
    from scripts.campaigns.r3_v53mp_run import _tier_pool_curve_ids

    dataset = _open_frozen_dataset(cfg)
    split_path = str((cfg.get("split_contract") or {}).get("split_manifest")
                     or "/pes2ts_data/v53mp_50k_data/split_manifest_v53mp.json")
    split_doc = json.loads(Path(split_path).read_text(encoding="utf-8"))
    pool_ids = _tier_pool_curve_ids(split_doc, "50k")
    _train_pos, val_pos = v53mp_pool_split(
        dataset.samples, pool_curve_ids=pool_ids, seed=int(seed))
    needed = int(n_batches) * int(positions_per_batch)
    if len(val_pos) < needed:
        raise V542ParityError(
            "V542_PARITY_BATCHES",
            f"inner-val slice for seed {seed} has {len(val_pos)} positions, "
            f"need >= {needed}")

    base = [int(p) for p in val_pos[:needed]]

    # --- category counts over the frozen base ----------------------------
    counts: Dict[str, int] = {c: 0 for c in REQUIRED_COVERAGE_CATEGORIES}
    for p in base:
        cats = _coverage_categories_of_record(dataset, dataset.samples[p])
        for c in cats & counts.keys():
            counts[c] += 1
    missing = [c for c in REQUIRED_COVERAGE_CATEGORIES if counts[c] == 0]

    # --- deterministic first-hit scan for every missing category ---------
    extra: List[int] = []
    if missing:
        still = set(missing)
        for p in val_pos[needed:]:
            rec = dataset.samples[int(p)]
            cats = _coverage_categories_of_record(dataset, rec)
            hit = sorted(still & cats)
            if not hit:
                continue
            pos = int(p)
            extra.append(pos)
            for c in hit:
                counts[c] += 1
                still.discard(c)
            if not still:
                break
        if still:
            raise V542ParityError(
                "V542_PARITY_COVERAGE",
                f"inner-val slice for seed {seed} lacks required coverage "
                f"categories {sorted(still)} (base={needed} positions "
                f"scanned; categories present in base="
                f"{sorted(c for c, n in counts.items() if n > 0)})")

    positions = base + extra
    batches: List[Dict[str, Any]] = []
    for b in range(int(n_batches)):
        chunk = base[b * positions_per_batch:(b + 1) * positions_per_batch]
        batches.append((chunk, None))
    if extra:
        batches.append((extra, None))

    lru = _FeatureLRU(dataset, capacity=len(positions) + 64)
    built: List[Dict[str, Any]] = []
    for chunk, _ in batches:
        batch, _records = _batch_from_positions(dataset, lru, chunk, "cpu")
        built.append(forward_view_v3(batch))

    coverage: Dict[str, Any] = {
        "required_categories": list(REQUIRED_COVERAGE_CATEGORIES),
        "measured_only_categories": list(MEASURED_ONLY_COVERAGE_CATEGORIES),
        "counts": {c: int(counts[c]) for c in REQUIRED_COVERAGE_CATEGORIES},
        "present": {c: bool(counts[c] > 0)
                    for c in REQUIRED_COVERAGE_CATEGORIES},
        "missing_required": [],
        "small_graph_max_atoms": SMALL_GRAPH_MAX_ATOMS,
        "large_graph_min_atoms": LARGE_GRAPH_MIN_ATOMS,
        "n_positions": len(positions),
        "n_base_positions": len(base),
        "n_coverage_positions": len(extra),
        "coverage_positions_appended": list(extra),
        "selection_rule": (
            "frozen first-8-chunks base + deterministic first-hit "
            "append per missing required category (sorted inner-val "
            "order; no RNG)"),
        "graph_structure": _graph_level_coverage_counts(built),
    }
    view = {"positions": positions, "batches": built, "coverage": coverage}
    _BATCH_CACHE[key] = view
    return view


def inner_val_batches(cfg: Mapping[str, Any], seed: int, *,
                      n_batches: int = MIN_INNER_VAL_BATCHES,
                      positions_per_batch: int = BATCH_POSITIONS,
                      ) -> List[Dict[str, Any]]:
    """``n_batches`` (+ coverage extras) REAL inner-val batches for one seed.

    Uses the frozen V542 data view exactly as the V541 trainer/evaluator
    does: 50K pool curve ids from the pinned split manifest →
    ``v53mp_pool_split(seed=…)`` curve-level 80/20 inner split → the
    coverage-aware deterministic selection of
    :func:`inner_val_selection` → ``_batch_from_positions`` (records →
    ``build_sample_features_v3`` → ``collate_v3`` → ``to_torch_batch``).
    Deterministic by construction (no RNG draw touches the batch slice).
    """
    return inner_val_selection(
        cfg, seed, n_batches=n_batches,
        positions_per_batch=positions_per_batch)["batches"]


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------
def build_pair(payload: Mapping[str, Any]
               ) -> tuple[V542Model, KeyAtomV53MPModel, Dict[str, Any]]:
    """Strict-audit-load BOTH models; RETURN the loader audit (gate-1 C1)."""
    state = payload["model"] if "model" in payload else payload
    legacy = KeyAtomV53MPModel()
    legacy.load_state_dict(state, strict=True)
    v542 = V542Model()
    audit = load_a0_strict(v542, {"model": state})
    assert v542.backbone.w_v is v542.move_decoder.w_v, \
        "shared-w_v invariant broken after load"
    return v542, legacy, audit


def _compare(reference: Mapping[str, torch.Tensor],
             candidate: Mapping[str, torch.Tensor],
             keys: Sequence[str]) -> Dict[str, Any]:
    report: Dict[str, Any] = {}
    all_bitwise = True
    for key in keys:
        a, b = reference[key], candidate[key]
        bitwise = bool(torch.equal(a, b))
        max_abs = float((a - b).abs().max().item()) if a.numel() else 0.0
        report[key] = {"bitwise": bitwise, "max_abs_diff": max_abs}
        all_bitwise = all_bitwise and bitwise
    report["all_bitwise"] = bool(all_bitwise)
    return report


def run_cpu_parity(v542: V542Model, legacy: KeyAtomV53MPModel,
                   batches: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """CPU bitwise parity: five outputs + forward_move + q is order_score."""
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        v542.eval()
        legacy.eval()
        per_batch: List[Dict[str, Any]] = []
        all_bitwise = True
        move_bitwise = True
        q_is_order = True
        with torch.no_grad():
            for batch in batches:
                out_legacy = legacy(batch)
                out_v542 = v542(batch)
                cmp5 = _compare(out_legacy, out_v542, FORWARD_KEYS_V542)
                mv_legacy = legacy.forward_move(batch)
                mv_v542 = v542.forward_move(batch)
                cmp_move = _compare(mv_legacy, mv_v542, ("u", "z"))
                qio = bool(out_v542["order_score"] is out_v542["q"]) and \
                    bool(torch.equal(out_legacy["q"], out_legacy["order_score"]))
                per_batch.append({
                    "outputs": cmp5,
                    "forward_move": cmp_move,
                    "q_is_order_score_v542": bool(out_v542["order_score"]
                                                  is out_v542["q"]),
                    "q_equals_order_score": qio,
                })
                all_bitwise = all_bitwise and cmp5["all_bitwise"]
                move_bitwise = move_bitwise and cmp_move["all_bitwise"]
                q_is_order = q_is_order and qio
        # aggregate max-abs per key across batches (decisive = bitwise flag)
        agg: Dict[str, Any] = {}
        for key in FORWARD_KEYS_V542:
            agg[key] = {
                "bitwise": all(pb["outputs"][key]["bitwise"] for pb in per_batch),
                "max_abs_diff": max(pb["outputs"][key]["max_abs_diff"]
                                    for pb in per_batch),
            }
        return {
            "n_batches": len(batches),
            "per_key": agg,
            "all_bitwise": bool(all_bitwise),
            "forward_move_bitwise": bool(move_bitwise),
            "q_is_order_score": bool(q_is_order),
            "threads": 1,
            "per_batch": per_batch,
        }
    finally:
        torch.set_num_threads(prev_threads)


def _to_device(batch: Mapping[str, Any], device: str) -> Dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in batch.items()}


def _gpu_floor(model: torch.nn.Module, batch: Mapping[str, Any],
               keys: Sequence[str]) -> Dict[str, Any]:
    """Run-to-run GPU floor: same model, same batch, two consecutive runs."""
    with torch.no_grad():
        first = model(batch)
        second = model(batch)
    floor: Dict[str, Any] = {}
    for key in keys:
        a, b = first[key], second[key]
        floor[key] = {
            "bitwise": bool(torch.equal(a, b)),
            "max_abs_diff": float((a - b).abs().max().item()) if a.numel() else 0.0,
            "within_cap": bool(torch.allclose(
                a, b, atol=GPU_ATOL, rtol=GPU_RTOL)),
        }
    floor["all_within_cap"] = all(v["within_cap"] for v in floor.values())
    return floor


def run_gpu_tolerance(v542: V542Model, legacy: KeyAtomV53MPModel,
                      batches: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """GPU tolerance parity with hard caps (atol 1e-6 / rtol 1e-5, TF32 off).

    The run-to-run floor is a DIAGNOSTIC: floor > cap ⇒ status
    ``inconclusive`` (CPU bitwise stays decisive); a real v542-vs-legacy
    breach with a clean floor ⇒ ``fail``.  Caps are never loosened.
    """
    if not torch.cuda.is_available():
        return {"status": "skipped_no_cuda",
                "atol": GPU_ATOL, "rtol": GPU_RTOL, "tf32": False}
    prev_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
    prev_tf32_cudnn = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        v542_gpu = v542.cuda().eval()
        legacy_gpu = legacy.cuda().eval()
        assert v542_gpu.backbone.w_v is v542_gpu.move_decoder.w_v, \
            "shared-w_v invariant broken on CUDA transfer"
        per_key_max: Dict[str, float] = {k: 0.0 for k in FORWARD_KEYS_V542}
        within = {k: True for k in FORWARD_KEYS_V542}
        floor_all_within = True
        floor_max = 0.0
        with torch.no_grad():
            for cpu_batch in batches:
                batch = _to_device(cpu_batch, "cuda")
                out_legacy = legacy_gpu(batch)
                out_v542 = v542_gpu(batch)
                for key in FORWARD_KEYS_V542:
                    diff = float((out_legacy[key] - out_v542[key])
                                 .abs().max().item())
                    per_key_max[key] = max(per_key_max[key], diff)
                    within[key] = within[key] and bool(torch.allclose(
                        out_legacy[key], out_v542[key],
                        atol=GPU_ATOL, rtol=GPU_RTOL))
                for model in (legacy_gpu, v542_gpu):
                    floor = _gpu_floor(model, batch, FORWARD_KEYS_V542)
                    floor_all_within = floor_all_within and bool(
                        floor["all_within_cap"])
                    for key in FORWARD_KEYS_V542:
                        floor_max = max(floor_max,
                                        floor[key]["max_abs_diff"])
        if all(within.values()):
            status = "pass"
        elif not floor_all_within:
            # The model cannot reproduce ITSELF within the cap — the GPU
            # comparison is meaningless; CPU bitwise remains decisive.
            status = "inconclusive"
        else:
            status = "fail"
        return {
            "status": status,
            "atol": GPU_ATOL,
            "rtol": GPU_RTOL,
            "tf32": False,
            "per_key_max_abs_diff": per_key_max,
            "per_key_within_cap": within,
            "run_to_run_floor_max_abs_diff": floor_max,
            "run_to_run_floor_all_within_cap": floor_all_within,
            "note": ("caps are hard (config thresholds.gpu_parity_*); "
                     "floor>cap => inconclusive, never a loosened pass"),
        }
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_tf32_matmul
        torch.backends.cudnn.allow_tf32 = prev_tf32_cudnn
        v542.cpu()
        legacy.cpu()


# ---------------------------------------------------------------------------
# evaluator parity leg (config parity.evaluator_parity =
# recompute_H2_H3_vs_V541_archive_bitwise_or_le_1e-9; ADR-0033 §2.2)
# ---------------------------------------------------------------------------
_EVALUATOR_CTX_CACHE: Dict[tuple, Dict[str, Any]] = {}


def _inner_val_split(cfg: Mapping[str, Any], seed: int
                     ) -> tuple[Any, List[int], List[int]]:
    """``(dataset, train_pos, val_pos)`` of the frozen 50K pool (cached).

    Shares the exact split derivation with :func:`inner_val_batches` (the
    SAME ``v53mp_pool_split`` curve-level 80/20 inner split) — no second
    split implementation exists here.
    """
    key = (str(cfg.get("campaign_id")), int(seed))
    if key in _EVALUATOR_CTX_CACHE:
        ctx = _EVALUATOR_CTX_CACHE[key]
        return ctx["dataset"], ctx["train_pos"], ctx["val_pos"]

    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        v53mp_pool_split,
    )
    from scripts.campaigns.r3_v53mp_run import _tier_pool_curve_ids

    dataset = _open_frozen_dataset(cfg)
    split_path = str((cfg.get("split_contract") or {}).get("split_manifest")
                     or "/pes2ts_data/v53mp_50k_data/split_manifest_v53mp.json")
    split_doc = json.loads(Path(split_path).read_text(encoding="utf-8"))
    pool_ids = _tier_pool_curve_ids(split_doc, "50k")
    train_pos, val_pos = v53mp_pool_split(
        dataset.samples, pool_curve_ids=pool_ids, seed=int(seed))
    return dataset, train_pos, val_pos


def evaluator_context(cfg: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    """Frozen evaluator inputs for one run seed (cached per process).

    Reuses the EXISTING data/label/cache constructions only — no new
    feature or label pipeline is built here:

    * dataset: the frozen 50K ``ColumnarKeyAtomDatasetV3`` (READ-ONLY);
    * ``labels_v52`` / ``endpoint_labels``: the dataset's lazy
      ``label_tables()`` — the same v52 + endpoint tables the campaign
      driver hands the frozen trainer/evaluator (v2-loader path of
      ``scripts/campaigns/r3_v53mp_run.py``);
    * ``cache``: ``_CacheView(_FeatureLRU(...))`` — the ``cache[position]
      -> built features`` protocol ``evaluate_v53mp`` /
      ``evaluate_checkpoint`` consume;
    * ``config``: ``V53MPTrainerConfig(arm="A", seed=seed, device="cpu")``
      (evaluation-only use: device + move-guard policy fields);
    * ``train_q_mean``: the frozen ``_train_q_mean`` over the INNER-TRAIN
      slice of the same split (identical for both models under test);
    * ``tau_end``: deliberately ``None`` — no tau calibration is invented
      by this certificate (tau-dependent misstop metrics are None on both
      sides and compare equal exactly).

    Typed failure: ``EVALUATOR_PARITY_UNAVAILABLE`` carrying the exact
    missing dependency when the label/feature access surface cannot be
    driven (never a silent skip).
    """
    key = (str(cfg.get("campaign_id")), int(seed))
    if key in _EVALUATOR_CTX_CACHE:
        return _EVALUATOR_CTX_CACHE[key]

    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        V53MPTrainerConfig,
        _CacheView,
        _FeatureLRU,
        _train_q_mean,
    )

    try:
        try:
            dataset, train_pos, val_pos = _inner_val_split(cfg, seed)
        except V542ParityError as exc:
            # shared data-view refusal => the evaluator leg reports its own
            # typed code while naming the exact missing dependency
            raise V542ParityError(
                "EVALUATOR_PARITY_UNAVAILABLE",
                f"frozen 50K data view unavailable for the evaluator leg "
                f"({exc.code}): {exc.detail}") from exc
        try:
            labels_v52, endpoint_labels = dataset.label_tables()
        except Exception as exc:  # noqa: BLE001 - typed conversion
            raise V542ParityError(
                "EVALUATOR_PARITY_UNAVAILABLE",
                f"frozen label tables unavailable via "
                f"ColumnarKeyAtomDatasetV3.label_tables(): {exc}") from exc
        # probe the evaluator's label access surface on a real inner-val
        # position (exact missing dependency named on failure)
        if not val_pos:
            raise V542ParityError(
                "EVALUATOR_PARITY_UNAVAILABLE",
                f"inner-val slice for seed {seed} is empty")
        probe_sid = str(dataset.samples[int(val_pos[0])].sample_id)
        if probe_sid not in labels_v52.by_sample:
            raise V542ParityError(
                "EVALUATOR_PARITY_UNAVAILABLE",
                f"v52 label table misses probe sample {probe_sid!r} "
                f"(seed {seed} inner-val position {int(val_pos[0])})")
        if probe_sid not in endpoint_labels.by_sample:
            raise V542ParityError(
                "EVALUATOR_PARITY_UNAVAILABLE",
                f"endpoint label table misses probe sample {probe_sid!r} "
                f"(seed {seed} inner-val position {int(val_pos[0])})")
    except V542ParityError:
        raise
    except Exception as exc:  # noqa: BLE001 - typed conversion
        raise V542ParityError(
            "EVALUATOR_PARITY_UNAVAILABLE",
            f"frozen 50K data view cannot drive the evaluator: {exc}"
        ) from exc

    # same inner-val positions as the parity batches (deterministic slice)
    positions = [int(p) for p in
                 val_pos[:MIN_INNER_VAL_BATCHES * BATCH_POSITIONS]]
    needed = MIN_INNER_VAL_BATCHES * BATCH_POSITIONS
    if len(positions) < needed:
        raise V542ParityError(
            "EVALUATOR_PARITY_UNAVAILABLE",
            f"inner-val slice for seed {seed} has {len(val_pos)} positions, "
            f"need >= {needed}")
    lru = _FeatureLRU(dataset, capacity=len(positions) + 64)
    ctx: Dict[str, Any] = {
        "dataset": dataset,
        "train_pos": list(train_pos),
        "val_pos": list(val_pos),
        "labels_v52": labels_v52,
        "endpoint_labels": endpoint_labels,
        "cache": _CacheView(lru),
        "lru": lru,
        "config": V53MPTrainerConfig(arm="A", seed=int(seed), device="cpu"),
        "positions": positions,
        "train_q_mean": _train_q_mean(dataset.samples, labels_v52, train_pos),
        "seed": int(seed),
    }
    _EVALUATOR_CTX_CACHE[key] = ctx
    return ctx


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def compare_evaluator_metrics(reference: Any, candidate: Any, *,
                              tol: float = EVALUATOR_ATOL) -> Dict[str, Any]:
    """Leaf-by-leaf comparator over two ``evaluate_v53mp`` outputs.

    Comparison surface (documented): EVERY leaf under the top-level keys
    ``EVALUATOR_TOP_KEYS`` of the frozen evaluator output — nested dicts
    and lists are walked recursively.

    Rules (never loosened):

    * float leaves: bitwise equality (``==``) OR ``|ref - cand| <= tol``
      (absolute, tol = ``EVALUATOR_ATOL`` = 1e-9; NaN == NaN passes);
    * int / bool / str / None leaves: EXACT equality (counts, masks and
      run_outcome are never within-tolerance);
    * structure: a key present on one side only, or a non-numeric type
      mismatch, is a mismatch (recorded with kind).

    Returns a report dict: ``all_equal``, ``max_abs_diff`` (max |ref-cand|
    over all numeric leaves), ``n_float_leaves``, ``n_exact_leaves``,
    ``mismatches`` (each: ``key`` path, ``kind``, ``reference``,
    ``candidate``, and ``abs_diff`` for numeric leaves).
    """
    mismatches: List[Dict[str, Any]] = []
    max_abs = 0.0
    n_float = 0
    n_exact = 0

    def _walk(path: str, ref: Any, cand: Any) -> None:
        nonlocal max_abs, n_float, n_exact
        if isinstance(ref, Mapping) and isinstance(cand, Mapping):
            for k in sorted(set(ref) | set(cand), key=str):
                p = f"{path}.{k}" if path else str(k)
                if k not in ref:
                    mismatches.append({"key": p, "kind": "missing_in_reference",
                                       "reference": None, "candidate": cand[k]})
                elif k not in cand:
                    mismatches.append({"key": p, "kind": "missing_in_candidate",
                                       "reference": ref[k], "candidate": None})
                else:
                    _walk(p, ref[k], cand[k])
            return
        if isinstance(ref, (list, tuple)) or isinstance(cand, (list, tuple)):
            if (isinstance(ref, (list, tuple)) and
                    isinstance(cand, (list, tuple)) and
                    len(ref) == len(cand)):
                for i, (a, b) in enumerate(zip(ref, cand)):
                    _walk(f"{path}[{i}]", a, b)
            else:
                mismatches.append({"key": path, "kind": "structure",
                                   "reference": ref, "candidate": cand})
            return
        if isinstance(ref, bool) or isinstance(cand, bool):
            n_exact += 1
            if not (isinstance(ref, bool) and isinstance(cand, bool)
                    and ref == cand):
                mismatches.append({"key": path, "kind": "exact",
                                   "reference": ref, "candidate": cand})
            return
        if isinstance(ref, int) and isinstance(cand, int):
            n_exact += 1
            if ref != cand:
                mismatches.append({"key": path, "kind": "exact",
                                   "reference": ref, "candidate": cand})
            return
        if _is_number(ref) and _is_number(cand):
            rf, cf = float(ref), float(cand)
            n_float += 1
            if math.isnan(rf) and math.isnan(cf):
                return  # NaN == NaN passes (both sides absent identically)
            diff = abs(rf - cf)
            max_abs = max(max_abs, diff)
            if not (rf == cf or diff <= tol):
                mismatches.append({"key": path, "kind": "float",
                                   "reference": ref, "candidate": cand,
                                   "abs_diff": diff})
            return
        n_exact += 1
        if type(ref) is not type(cand) or ref != cand:
            mismatches.append({"key": path, "kind": "exact",
                               "reference": ref, "candidate": cand})

    _walk("", reference, candidate)
    return {
        "tolerance": float(tol),
        "compared_top_keys": list(EVALUATOR_TOP_KEYS),
        "n_float_leaves": n_float,
        "n_exact_leaves": n_exact,
        "max_abs_diff": float(max_abs),
        "mismatches": mismatches,
        "all_equal": not mismatches,
    }


def _evaluator_noop_problems(reference: Any, candidate: Any,
                             n_float_leaves: int) -> List[str]:
    """Gate-1 A7: why an evaluator comparison is a no-op (empty list = real).

    A future no-op evaluator returning ``{}`` (or missing the documented
    top-level keys) on both sides would compare "equal" vacuously — the
    real evaluator leg must fail instead.
    """
    problems: List[str] = []
    for side_name, metrics in (("reference", reference),
                               ("candidate", candidate)):
        if not isinstance(metrics, Mapping):
            problems.append(
                f"{side_name} evaluator output is not a mapping "
                f"({type(metrics).__name__})")
            continue
        missing_top = [k for k in EVALUATOR_TOP_KEYS if k not in metrics]
        if missing_top:
            problems.append(
                f"{side_name} evaluator output misses expected top-level "
                f"keys {missing_top}")
    if int(n_float_leaves) <= 0:
        problems.append(
            "n_float_leaves == 0 — the evaluator comparison has no float "
            "metric leaf (no-op evaluator cannot certify)")
    return problems


def run_evaluator_parity(v542: V542Model, legacy: KeyAtomV53MPModel,
                         cfg: Mapping[str, Any], seed: int) -> Dict[str, Any]:
    """Evaluator leg: frozen ``evaluate_v53mp`` on v542 vs legacy.

    Runs the SAME frozen evaluator over the SAME inner-val positions,
    label tables, feature cache and config twice — once per model — and
    compares every metric leaf bitwise-or-1e-9 (see
    :func:`compare_evaluator_metrics`).  Deterministic by construction
    (fixed bootstrap seeds inside the frozen evaluator; CPU threads=1).

    Returns the per-asset evaluator report (``status`` pass/fail).  A
    mismatch is a HARD fail: the report carries the first offending key +
    both values + the overall max abs delta.
    """
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        evaluate_v53mp,
    )

    ctx = evaluator_context(cfg, seed)
    prev_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        v542.eval()
        legacy.eval()
        with torch.no_grad():
            metrics_legacy = evaluate_v53mp(
                legacy, None, ctx["dataset"], ctx["labels_v52"],
                ctx["endpoint_labels"], ctx["cache"], ctx["positions"],
                ctx["config"], train_q_mean=ctx["train_q_mean"], tau_end=None)
            metrics_v542 = evaluate_v53mp(
                v542, None, ctx["dataset"], ctx["labels_v52"],
                ctx["endpoint_labels"], ctx["cache"], ctx["positions"],
                ctx["config"], train_q_mean=ctx["train_q_mean"], tau_end=None)
        comparison = compare_evaluator_metrics(metrics_legacy, metrics_v542)
    finally:
        torch.set_num_threads(prev_threads)
        # evaluate_v53mp leaves the model in train(True) — restore eval
        v542.eval()
        legacy.eval()

    # gate-1 A7: a no-op evaluator (missing expected top-level keys, or
    # zero float leaves) can NEVER pass — the comparison must have real
    # metric surface on both sides.
    noop_problems = _evaluator_noop_problems(
        metrics_legacy, metrics_v542, comparison["n_float_leaves"])

    status = "pass" if comparison["all_equal"] else "fail"
    error = None
    if noop_problems:
        status = "fail"
        error = ("evaluator NO-OP guard (gate-1 A7): "
                 + "; ".join(noop_problems))
    elif not comparison["all_equal"]:
        first = comparison["mismatches"][0]
        error = (
            "evaluator parity BREAK: key={key} kind={kind} "
            "reference={reference!r} candidate={candidate!r} "
            "abs_diff={abs_diff!r} n_mismatches={n} max_abs_diff={mx!r} "
            "(tolerance {tol})".format(
                key=first.get("key"), kind=first.get("kind"),
                reference=first.get("reference"),
                candidate=first.get("candidate"),
                abs_diff=first.get("abs_diff"),
                n=len(comparison["mismatches"]),
                mx=comparison["max_abs_diff"], tol=EVALUATOR_ATOL))
    return {
        "status": status,
        "rule": ("recompute_H2_H3_vs_V541_archive_bitwise_or_le_1e-9 "
                 "(config parity.evaluator_parity; executed as frozen-"
                 "evaluator recomputation on v542 vs legacy reference, "
                 "same positions/labels/cache/config)"),
        "evaluator": ("pes2ts_core.pesn.offline.training."
                      "key_atom_v53mp_train_v3.evaluate_v53mp (frozen)"),
        "tolerance": EVALUATOR_ATOL,
        "compared_top_keys": list(EVALUATOR_TOP_KEYS),
        "positions_source": ("first inner-val positions of the seed's "
                             "v53mp_pool_split slice (the same frozen base "
                             "slice the parity batches start from; the parity "
                             "batches may append coverage positions beyond "
                             "this base — see real_batch_coverage)"),
        "n_positions": len(ctx["positions"]),
        "tau_end": None,
        "r3_baseline": None,
        "train_q_mean": ctx["train_q_mean"],
        "n_float_leaves": int(comparison["n_float_leaves"]),
        "n_exact_leaves": int(comparison["n_exact_leaves"]),
        "comparison": comparison,
        "error": error,
    }


# ---------------------------------------------------------------------------
# per-asset + full report
# ---------------------------------------------------------------------------
def _asset_load_audit_record(asset: Mapping[str, Any], path: Path,
                             audit: Mapping[str, Any]) -> Dict[str, Any]:
    """Per-asset strict-load audit record for the certificate (gate-1 C1).

    Carries EVERY field the certificate template's ``load_audit.record``
    requires — ``source_sha256`` (the ACTUAL verified file digest),
    ``mapping_table_sha256`` (the CANONICAL digest; gate-0 advisory A3
    closed — same frozen recipe as every V5.4.2 identity field),
    ``target_key_set`` (the 45 mapped target keys), ``audit_json_path``
    (null: the audit is embedded in THIS report — the report path is the
    audit path; exact mapping declared in the report-level
    ``template_field_mapping``) — PLUS the full loader audit surface
    (43 source keys / 45 mapped targets / dual-write aliases /
    torch_equal count / empty failure categories).
    """
    return {
        "ok": bool(audit.get("ok")),
        "strict": True,
        "schema": str(audit.get("schema", "")),
        "source_path": str(asset["path"]),
        "source_sha256": sha256_file(path),
        "mapping_table_sha256": str(audit.get("mapping_table_sha256", "")),
        "n_source_keys": int(audit.get("n_source_keys", 0)),
        "n_mapped_targets": int(audit.get("n_mapped_targets", 0)),
        "dual_write_aliases": dict(audit.get("dual_write_aliases") or {}),
        "target_key_set": sorted(audit.get("loaded_keys") or []),
        "torch_equal_verified": len(audit.get("torch_equal_verified") or []),
        "missing": list(audit.get("missing") or []),
        "unexpected": list(audit.get("unexpected") or []),
        "shape_mismatch": list(audit.get("shape_mismatch") or []),
        "dtype_mismatch": list(audit.get("dtype_mismatch") or []),
        "audit_json_path": None,
        "audit_json_path_mapping": (
            "this per-asset audit record IS embedded in the parity report "
            "JSON written via --out (schema pes2ts_v542_a0_parity_report_v1); "
            "template load_audit.record.audit_json_path maps to that report "
            "path — no separate audit file is ever written"),
    }


def run_asset(asset: Mapping[str, Any], path: Path, cfg: Mapping[str, Any],
              *, skip_gpu: bool = False) -> Dict[str, Any]:
    """Full certificate for ONE pinned checkpoint (never raises)."""
    entry: Dict[str, Any] = {
        "id": str(asset["id"]),
        "path": str(asset["path"]),
        "source_tier": str(asset["source_tier"]),
        "training_updates": int(asset["training_updates"]),
        "sha256_pinned": str(asset["sha256"]),
        "sha256_actual": sha256_file(path),
        "passed": False,
        "error": None,
    }
    try:
        # trusted repo artifact — numpy RNG globals in the payload require
        # weights_only=False (same convention as every trainer loader here)
        payload = torch.load(str(path), map_location="cpu",
                             weights_only=False)
        if isinstance(payload, Mapping) and "model" in payload:
            entry["payload_schema"] = str(payload.get("schema", ""))
            entry["payload_update"] = payload.get("update")
        else:
            payload = {"model": payload}
        if not isinstance(payload.get("model"), Mapping):
            raise V542ParityError(
                "V542_PARITY_PAYLOAD",
                f"{path} has no mapping 'model' entry")

        v542, legacy, load_audit = build_pair(payload)
        # gate-1 C1: the FULL loader audit record is persisted per asset
        # (not a hardcoded {"ok": True} stub).
        entry["load_audit"] = _asset_load_audit_record(asset, path, load_audit)
        entry["n_parameters"] = {
            "v542_joint": v542.n_parameters,
            "legacy_joint": legacy.n_parameters,
            "v542_report": v542.parameter_report(),
        }

        seed = _seed_of(str(asset["id"]))
        selection = inner_val_selection(cfg, seed)
        batches = selection["batches"]
        entry["data_view"] = {
            "source": ("ColumnarKeyAtomDatasetV3 + v53mp_pool_split + "
                       "_batch_from_positions + forward_view_v3"),
            "split": "inner_val",
            "seed": int(seed),
            "n_batches": len(batches),
            "min_batches_frozen": MIN_INNER_VAL_BATCHES,
        }
        # gate-1 C3: actual per-category coverage of the selected batches.
        entry["real_batch_coverage"] = selection["coverage"]

        entry["cpu"] = run_cpu_parity(v542, legacy, batches)
        if skip_gpu:
            entry["gpu"] = {"status": "skipped_by_flag",
                            "atol": GPU_ATOL, "rtol": GPU_RTOL}
        else:
            entry["gpu"] = run_gpu_tolerance(v542, legacy, batches)

        # evaluator parity leg (typed unavailable => raises; mismatch =>
        # hard fail via status, never a silent skip)
        entry["evaluator"] = run_evaluator_parity(v542, legacy, cfg, seed)

        gpu = entry["gpu"]
        gpu_ok = gpu["status"] in ("pass", "inconclusive", "skipped_no_cuda",
                                   "skipped_by_flag")
        entry["passed"] = bool(
            entry["load_audit"]["ok"]
            and entry["cpu"]["all_bitwise"]
            and entry["cpu"]["forward_move_bitwise"]
            and entry["cpu"]["q_is_order_score"]
            and entry["evaluator"]["status"] == "pass"
            and gpu_ok)
        if not entry["passed"]:
            entry["error"] = (
                entry["evaluator"].get("error")
                if entry.get("evaluator", {}).get("error")
                else "parity criteria not met "
                     "(see cpu/gpu/evaluator report fields)")
    except Exception as exc:  # noqa: BLE001 - typed entry failure, no raise
        entry["error"] = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, V542ParityError):
            entry["error_code"] = exc.code
    return entry


def _gpu_disposition(entries: Sequence[Mapping[str, Any]],
                     *, skip_gpu: bool) -> str:
    """Aggregate GPU disposition over the ran entries (gate-1 C2).

    ``skipped_by_flag`` when ``--skip-gpu``; else ``fail`` if ANY entry
    failed the GPU leg; ``pass`` if every entry status is ``pass``;
    otherwise ``inconclusive`` (some entry inconclusive / no CUDA).
    """
    if skip_gpu:
        return "skipped_by_flag"
    statuses = [(e.get("gpu") or {}).get("status") for e in entries]
    if any(s == "fail" for s in statuses):
        return "fail"
    if statuses and all(s == "pass" for s in statuses):
        return "pass"
    return "inconclusive"


def run_parity(cfg: Mapping[str, Any], assets: Sequence[Mapping[str, Any]],
               paths: Mapping[str, Path], *,
               only: Optional[Sequence[str]] = None,
               skip_gpu: bool = False) -> Dict[str, Any]:
    """Audit every (required) asset; the report is strictly fail-closed.

    Gate-1 C2: the report distinguishes a full six-asset CERTIFICATE from
    a partial DIAGNOSTIC run.  ``certificate_complete`` is true only when
    all six required assets ran with no ``--asset`` filter and no
    ``--skip-gpu``; ``passed`` (certificate PASS) additionally requires
    every required entry passed AND the GPU disposition is ``pass`` or
    ``inconclusive``.  Partial runs are never certificates.
    """
    selected = [a for a in assets
                if only is None or str(a["id"]) in set(only)]
    if only is not None and len(selected) != len(set(only)):
        found = {str(a["id"]) for a in selected}
        raise V542ParityError(
            "V542_PARITY_ASSET_ID",
            f"unknown --asset ids: {sorted(set(only) - found)}")
    entries = [run_asset(a, paths[str(a["id"])], cfg, skip_gpu=skip_gpu)
               for a in selected]
    required = [e for e, a in zip(entries, selected) if a["required"] is True]
    gpu_disposition = _gpu_disposition(entries, skip_gpu=skip_gpu)
    # Gate-1 C2: a certificate needs the full six-asset surface with the
    # GPU leg executed (or honestly inconclusive) — never a filtered run.
    all_six_required_ran = (
        only is None and not skip_gpu
        and len(required) == 6
        and {e["id"] for e in required} == {str(a["id"]) for a in assets
                                            if a["required"] is True})
    certificate_complete = bool(all_six_required_ran)
    certificate_passed = bool(
        certificate_complete
        and required and all(e["passed"] for e in required)
        and gpu_disposition in ("pass", "inconclusive"))
    return {
        "schema": REPORT_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "certificate_scope": (
            "forward + evaluator-migration parity ONLY — NOT evidence of "
            "E0 training-comparability reuse (ADR-0033 §2.2 "
            "retrain_a0_x3).  Forward: 5 outputs + forward_move bitwise on "
            "real inner-val batches; evaluator: frozen evaluate_v53mp "
            "recomputed on v542 vs legacy reference, bitwise or <=1e-9 "
            "absolute on every metric leaf."),
        "report_schema_note": {
            "report_schema": REPORT_SCHEMA,
            "certificate_template_schema": "pes2ts_v542_parity_certificate_v1",
            "relationship": (
                "this report is the machine-produced EVIDENCE artifact; the "
                "template pes2ts_v542_parity_certificate_v1 is the signed "
                "certificate INSTANCE derived from it — the report is a "
                "superset (per-asset strict-load audit, coverage, "
                "certificate-completeness fields) and never silently "
                "diverges: every load_audit.record field the template "
                "requires is either present verbatim in each entry's "
                "load_audit or named in template_field_mapping below"),
            "template_field_mapping": {
                "source_sha256": ("entries[].load_audit.source_sha256 "
                                  "(actual verified file digest)"),
                "mapping_table_sha256": ("entries[].load_audit."
                                         "mapping_table_sha256 "
                                         "(canonical SHA-256 of the "
                                         "mapping table via the frozen "
                                         "V5.4.2 identity recipe; gate-0 "
                                         "advisory A3 closed)"),
                "target_key_set": ("entries[].load_audit.target_key_set "
                                   "(45 mapped target keys)"),
                "audit_json_path": ("entries[].load_audit."
                                    "audit_json_path — null; the per-asset "
                                    "audit record is embedded in this "
                                    "report JSON (see "
                                    "audit_json_path_mapping)"),
            },
        },
        "model_versions": {
            "v542": str(V542Model.v542_version),
            "legacy_reference": "KeyAtomV53MPModel (StarEGNNV3 + "
                                "ReactionProcessHeadV53MP)",
        },
        "tolerance": {
            "cpu": "torch.equal bitwise (decisive)",
            "gpu_atol": GPU_ATOL,
            "gpu_rtol": GPU_RTOL,
            "tf32_during_parity": False,
            "evaluator_atol": EVALUATOR_ATOL,
            "note": ("GPU caps are hard; run-to-run floor is diagnostic "
                     "only (floor>cap => inconclusive, CPU decisive).  "
                     "Evaluator metric leaves: bitwise or <=1e-9 absolute "
                     "(floats); ints/bools/strings/None exact."),
        },
        "evaluator_coverage": {
            "assets_covered": [e["id"] for e in entries
                               if "evaluator" in e],
            "assets_unavailable": [
                {"id": e["id"],
                 "error_code": e.get("error_code"),
                 "detail": e.get("error")}
                for e in entries if "evaluator" not in e],
            "note": ("all six pinned assets (both source tiers) share the "
                     "same KeyAtomV53MPModel/V542Model pair and the same "
                     "frozen evaluate_v53mp; the leg's only hard "
                     "dependency is the frozen 50K data view — absence is "
                     "a TYPED EVALUATOR_PARITY_UNAVAILABLE failure, never "
                     "a silent skip"),
        },
        "n_required_assets": len(required),
        "certificate_complete": certificate_complete,
        "only": list(only) if only is not None else None,
        "skip_gpu": bool(skip_gpu),
        "gpu_disposition": gpu_disposition,
        "entries": entries,
        "passed": certificate_passed,
        "passed_meaning": (
            "certificate PASS: all six required assets ran "
            "(certificate_complete), every required entry passed (CPU "
            "bitwise + evaluator + load audit) and gpu_disposition is "
            "pass/inconclusive.  A partial (--asset/--skip-gpu) run can "
            "never set passed=true."),
    }


def _main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="V5.4.2 A0 strict-load + forward parity certificate "
                    "(read-only over the six pinned checkpoints; writes "
                    "ONE new JSON report when --out is given)")
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help="campaign config carrying parity.assets "
                             "(default: %(default)s)")
    parser.add_argument("--asset", action="append", default=None,
                        metavar="ID", help="restrict to a pinned asset id "
                                           "(repeatable; all six by default). "
                                           "A filtered run is DIAGNOSTIC "
                                           "(not a certificate) and exits "
                                           "non-zero even when every selected "
                                           "asset passes")
    parser.add_argument("--skip-gpu", action="store_true",
                        help="skip the GPU tolerance leg (CPU bitwise stays "
                             "decisive).  The run is DIAGNOSTIC (not a "
                             "certificate) and exits non-zero even when all "
                             "assets pass")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="NEW path for the JSON report (refused if it "
                             "exists, pass --force to overwrite)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing --out report")
    args = parser.parse_args(argv)

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    try:
        cfg, assets = load_pinned_assets(config_path)
        paths = verify_assets(assets)   # all six verified even with --asset
    except V542ParityError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    try:
        report = run_parity(cfg, assets, paths, only=args.asset,
                            skip_gpu=args.skip_gpu)
    except V542ParityError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED

    if args.out:
        out_path = Path(args.out)
        if out_path.exists() and not args.force:
            print(f"REFUSED: output report already exists: {out_path} "
                  f"(pass --force to overwrite)", file=sys.stderr)
            return EXIT_REFUSED
        if out_path.is_dir():
            print(f"REFUSED: --out is a directory: {out_path}", file=sys.stderr)
            return EXIT_REFUSED
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8")

    for entry in report["entries"]:
        cpu = entry.get("cpu") or {}
        gpu = entry.get("gpu") or {}
        ev = entry.get("evaluator") or {}
        ev_cmp = ev.get("comparison") or {}
        status = "PASS" if entry["passed"] else "FAIL"
        print(f"[{status}] {entry['id']} tier={entry['source_tier']} "
              f"updates={entry['training_updates']} "
              f"cpu_bitwise={cpu.get('all_bitwise')} "
              f"move_bitwise={cpu.get('forward_move_bitwise')} "
              f"gpu={gpu.get('status')} "
              f"evaluator={ev.get('status', 'unavailable')} "
              f"eval_max_abs_delta={ev_cmp.get('max_abs_diff', 'n/a')}"
              + (f" — {entry['error']}" if entry.get("error") else ""))
    n_passed = sum(1 for e in report["entries"] if e["passed"])
    if not report["certificate_complete"]:
        # gate-1 C2: partial runs never certify — print an unmistakable
        # DIAGNOSTIC overall line and exit non-zero even on full asset
        # success (the CLI has no silent certificate mode).
        reasons = []
        if report["only"] is not None:
            reasons.append(f"--asset filter only={report['only']}")
        if report["skip_gpu"]:
            reasons.append("--skip-gpu")
        print(f"overall: DIAGNOSTIC (not a certificate) — "
              f"{n_passed}/{len(report['entries'])} selected assets passed; "
              f"certificate requires all 6 required assets with the GPU leg "
              f"({'; '.join(reasons) or 'incomplete asset set'})")
        return EXIT_FAIL
    print(f"overall: {'PASS' if report['passed'] else 'FAIL'} "
          f"({n_passed}/{len(report['entries'])} assets)"
          f" gpu_disposition={report['gpu_disposition']}")
    return EXIT_PASS if report["passed"] else EXIT_FAIL


def main(argv: Optional[Sequence[str]] = None) -> int:
    return _main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
