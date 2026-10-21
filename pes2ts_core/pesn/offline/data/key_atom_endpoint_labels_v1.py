"""V5.2-R D0 endpoint-stop labels: schema ``pes2ts_fullpath_nav_labels_v1``.

Implements the Endpoint Stop half of the frozen task contract
(``docs/contracts/PES2TS_FullPathNavigationContract_v1.md`` §5 + §7; ADR-0030;
V5.2-R plan §2).  The Move half (``move_status``/``move_target_u``) is
inherited verbatim from the R3 full-traversal teacher via
``SampleRecordV3.label_status``; this module only adds the geometric
endpoint-stop side:

* ``d_end = RMSD_K(X, X_target_endpoint)`` with Kabsch **proper-rotation**
  alignment over the intent's key-atom set K (:func:`kabsch_rmsd_v52`);
* target endpoint: **forward → B (P)** = last node of the canonical A→B path
  order; **reverse → A (R)** = first node (contract §5.1);
* three-state window label (contract §5.2): ``d_end <= w_pos`` → 1,
  ``d_end >= w_neg`` → 0, middle → −1 (Mask); windows inclusive at both
  edges; primary candidate window ``w_pos=0.10 Å / w_neg=0.30 Å``.

Core bug fix versus legacy ``pes2ts_e4_labels_v1`` (V5.2 task-correction
audit 20261021 §1): **Stop eligibility is geometric and independent of Move
eligibility** — ``terminal`` (no_future_contact), ``masked`` and ``conflict``
rows MUST receive a ``d_end``/label as well, so the arriving-endpoint
supervision that the old ``label_statuses=("ok",)`` training entry point
dropped is now visible.  :func:`build_endpoint_stop_labels` therefore
refuses (typed error) to build from a dataset load that did not include the
``terminal`` status — that exclusion is precisely the defect being fixed.

The two known TS/endpoint geometric degeneracies (``RXN_0000001174``
d_end(TS)=0.066 Å, ``RXN_0000001675`` d_end(TS)=0.019 Å; contract §5.3 rule
4) are flagged via :data:`DEGENERATE_TS_CURVES` so audits can report the
with/without-degenerate calibers; flagging never silently masks.

Read-only reuse: depends on ``key_atom_dataset_v3`` (KeyAtomDatasetV3,
V52LabelTable node orders, kabsch_rmsd_v52) without modifying it.  numpy-only.

V5.3-MP extension (plan §4.3 + MoveProcessContract v1 §5.3/§10.1) — this
module ALSO hosts the executable ``endpoint_label_spec_v1`` generator
(:func:`build_endpoint_labels_spec_v1` + :class:`EndpointLabelTableV1`),
which supersedes the three-way window semantics above for NEW training
(MOVE+PROCESS campaign): positive/negative/Mask with bond-edit completion,
geometry legality, TS hard-negatives, degenerate-curve masking and a
coverage report.  The V5.2-R table above stays as the frozen diagnostic
caliber; nothing about it changed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.offline.data.key_atom_dataset_v3 import (
    KeyAtomDatasetV3,
    V52LabelTable,
    kabsch_rmsd_v52,
)

#: Label-table schema id (contract §7; run-record ``label_schema``).
SCHEMA: Final[str] = "pes2ts_fullpath_nav_labels_v1"

#: Primary candidate windows (contract §5.2; candidate until D1 freezes one).
ENDPOINT_STOP_POSITIVE_WINDOW_ANGSTROM: Final[float] = 0.10
ENDPOINT_STOP_NEGATIVE_WINDOW_ANGSTROM: Final[float] = 0.30

#: TS/endpoint geometric degeneracies that MUST be flagged (contract §5.3
#: rule 4; audit 20261021 §6).  Flagged rows keep their geometric label —
#: reports carry the with/without-degenerate calibers instead of masking.
DEGENERATE_TS_CURVES: Final[tuple[str, ...]] = (
    "RXN_0000001174", "RXN_0000001675")

#: ``label_status`` (R3 Move teacher) → ``move_status`` of schema v1
#: (contract §7; ``terminal`` is the historical column name for
#: ``no_future_contact``).
MOVE_STATUS_MAP: Final[dict[str, str]] = {
    "ok": "ok",
    "terminal": "no_future_contact",
    "masked": "masked",
    "conflict": "conflict",
}


class EndpointStopLabelError(RuntimeError):
    """Typed refusal of the endpoint-stop label builder."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _require(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise EndpointStopLabelError(code, detail)


def endpoint_stop_label_from_distance(
    d_end: float,
    *,
    positive_window: float = ENDPOINT_STOP_POSITIVE_WINDOW_ANGSTROM,
    negative_window: float = ENDPOINT_STOP_NEGATIVE_WINDOW_ANGSTROM,
) -> int:
    """Frozen three-way endpoint-stop label: 1 / 0 / −1 (Mask).

    ``d_end <= positive_window`` ⇒ 1; ``d_end >= negative_window`` ⇒ 0;
    otherwise the middle region is MASKED (−1).  Windows are inclusive at
    both edges (contract §5.2).  Widening a window is a contract change,
    not a knob; an inverted order is a typed refusal.
    """
    if not (float(positive_window) <= float(negative_window)):
        raise EndpointStopLabelError(
            "ENDPOINT_WINDOW_ORDER",
            f"positive window {positive_window} > negative "
            f"{negative_window}")
    d = float(d_end)
    if d <= float(positive_window):
        return 1
    if d >= float(negative_window):
        return 0
    return -1


def target_endpoint_node(curve_labels_or_order, direction: str) -> int:
    """Node index of the intent's TARGET endpoint (contract §5.1).

    Accepts a ``CurveV52Labels`` (uses its ``node_order``) or any sequence
    of node indices in canonical A→B path order.  ``forward`` targets the
    path END (B / P); ``reverse`` targets the path START (A / R).
    """
    order: Sequence[int]
    node_order = getattr(curve_labels_or_order, "node_order", None)
    if node_order is not None:
        order = node_order
    else:
        order = curve_labels_or_order
    if direction == "forward":
        return int(order[-1])
    if direction == "reverse":
        return int(order[0])
    raise EndpointStopLabelError(
        "ENDPOINT_DIRECTION", f"direction must be forward|reverse, got {direction!r}")


@dataclass(frozen=True)
class EndpointStopSampleLabel:
    """One sample row's endpoint-stop label (schema v1 Stop side)."""

    sample_id: str
    curve_id: str
    node_index: int
    intent_id: str
    direction: str                     # "forward" | "reverse"
    target_endpoint: str               # "A" (R) | "B" (P)
    target_endpoint_node: int
    d_end_angstrom: float
    #: 1 (within w_pos of target endpoint) / 0 (≥ w_neg) / −1 (Mask)
    endpoint_stop_label: int
    #: True when this row is a TS frame of a contract §5.3 rule-4 curve
    degenerate_flag: bool
    #: Move eligibility inherited from R3 (contract §7 move_status enum)
    move_status: str

    @property
    def is_start_row(self) -> bool:
        """True when the row sits on the OPPOSITE (start) endpoint."""
        return self.direction == "forward" and self.target_endpoint == "A" \
            or self.direction == "reverse" and self.target_endpoint == "B"

    def stop_status(self) -> str:
        """Contract §7 ``stop_status`` derived from label + degeneracy.

        Degenerate TS rows (rule 4) map to ``invalid`` regardless of their
        geometric label so the with/without-degenerate calibers are
        reproducible from the table alone.
        """
        if self.degenerate_flag:
            return "invalid"
        if self.endpoint_stop_label == 1:
            return "endpoint_pos"
        if self.endpoint_stop_label == 0:
            return "endpoint_neg"
        return "endpoint_ambiguous"


@dataclass(frozen=True)
class CurveEndpointStopSummary:
    """Per-curve roll-up of the endpoint-stop label table."""

    curve_id: str
    endpoint_a_node: int
    endpoint_b_node: int
    ts_index: int
    n_rows: int
    n_pos: int
    n_neg: int
    n_mask: int
    n_degenerate: int
    d_end_min: float
    d_end_median: float
    d_end_max: float


@dataclass(frozen=True)
class EndpointStopLabelTable:
    """Endpoint-stop labels over a FULL-status dataset load.

    ``by_sample`` is keyed by ``sample_id`` and covers every loaded row —
    including ``terminal``/``masked``/``conflict`` rows (eligibility
    independence, contract §5.3 rule 1).
    """

    by_sample: Mapping[str, EndpointStopSampleLabel]
    curves: Mapping[str, CurveEndpointStopSummary]
    positive_window: float
    negative_window: float

    @property
    def n_rows(self) -> int:
        return len(self.by_sample)

    def count_label(self, label: int) -> int:
        return sum(1 for lab in self.by_sample.values()
                   if lab.endpoint_stop_label == label)

    @property
    def n_positive(self) -> int:
        return self.count_label(1)

    @property
    def n_negative(self) -> int:
        return self.count_label(0)

    @property
    def n_mask(self) -> int:
        return self.count_label(-1)

    def count_move_status(self, move_status: str) -> int:
        return sum(1 for lab in self.by_sample.values()
                   if lab.move_status == move_status)

    def count_terminal_rows(self) -> int:
        return self.count_move_status("no_future_contact")

    def positive_counts_by_curve(self) -> dict[str, int]:
        out: dict[str, int] = {cid: 0 for cid in self.curves}
        for lab in self.by_sample.values():
            if lab.endpoint_stop_label == 1:
                out[lab.curve_id] += 1
        return out

    def to_rows(self) -> list[dict]:
        """Flat row dicts (sorted by sample_id) for audit writers."""
        rows: list[dict] = []
        for sid in sorted(self.by_sample):
            lab = self.by_sample[sid]
            rows.append({
                "sample_id": lab.sample_id,
                "curve_id": lab.curve_id,
                "node_index": lab.node_index,
                "intent_id": lab.intent_id,
                "direction": lab.direction,
                "target_endpoint": lab.target_endpoint,
                "target_endpoint_node": lab.target_endpoint_node,
                "d_end_angstrom": lab.d_end_angstrom,
                "endpoint_stop_label": lab.endpoint_stop_label,
                "degenerate_flag": lab.degenerate_flag,
                "move_status": lab.move_status,
            })
        return rows

    def as_dict(self) -> dict:
        """JSON-ready summary + rows (schema-stamped, deterministic)."""
        return {
            "schema": SCHEMA,
            "positive_window_angstrom": self.positive_window,
            "negative_window_angstrom": self.negative_window,
            "n_rows": self.n_rows,
            "n_positive": self.n_positive,
            "n_negative": self.n_negative,
            "n_mask": self.n_mask,
            "move_status_counts": {
                status: self.count_move_status(status)
                for status in ("ok", "no_future_contact", "masked",
                               "conflict")},
            "curves": [self.curves[cid].__dict__ for cid in sorted(self.curves)],
            "rows": self.to_rows(),
        }


def build_endpoint_stop_labels(
    dataset: KeyAtomDatasetV3,
    v52_labels: V52LabelTable,
    *,
    positive_window: float = ENDPOINT_STOP_POSITIVE_WINDOW_ANGSTROM,
    negative_window: float = ENDPOINT_STOP_NEGATIVE_WINDOW_ANGSTROM,
) -> EndpointStopLabelTable:
    """Build the ``pes2ts_fullpath_nav_labels_v1`` endpoint-stop table.

    Caller contract: ``dataset`` must be loaded with ALL label statuses
    (``label_statuses=("ok", "terminal", "masked", "conflict")``) — the
    builder refuses a load that excludes ``terminal`` because dropping
    no_future_contact rows from Stop data is exactly the V5.2 defect this
    schema fixes (contract §5.3 rule 1; audit 20261021 §1).  ``v52_labels``
    supplies the canonical A→B ``node_order`` per curve (from
    ``KeyAtomDatasetV3.v52_labels``); it must cover the same sample rows.

    Per row: ``K = intent.key_rows()``; ``d_end = kabsch_rmsd_v52`` of the
    current frame's K geometry against the target-endpoint frame's K
    geometry (proper rotation only).  ``d_end`` is cached per
    ``(curve, node, target_node)`` — forward/reverse intents of one curve
    share the K set, so the mirrored intent reuses the cached value.
    """
    _require("terminal" in dataset.label_statuses, "TERMINAL_NOT_LOADED",
             "dataset was loaded without the 'terminal' status; endpoint-stop "
             "labels require the FULL status set (ok/terminal/masked/conflict) "
             "so no_future_contact rows enter Stop data (contract §5.3 rule 1)")
    if not (float(positive_window) <= float(negative_window)):
        raise EndpointStopLabelError(
            "ENDPOINT_WINDOW_ORDER",
            f"positive window {positive_window} > negative "
            f"{negative_window}")

    loaded_ids = {rec.sample_id for rec in dataset.samples}
    missing = loaded_ids - set(v52_labels.by_sample)
    _require(not missing, "V52_LABEL_COVERAGE",
             f"v52_labels misses {len(missing)} loaded sample rows "
             f"(e.g. {sorted(missing)[:3]}); rebuild via dataset.v52_labels() "
             "with the SAME filters")

    labels: dict[str, EndpointStopSampleLabel] = {}
    curve_rows: dict[str, list[EndpointStopSampleLabel]] = {}
    # d_end cache: (curve_id, node_index, target_node) → float
    d_end_cache: dict[tuple[str, int, int], float] = {}

    for rec in dataset.samples:
        cid = rec.curve_id
        curve_labels = v52_labels.curves.get(cid)
        _require(curve_labels is not None, "V52_CURVE_COVERAGE",
                 f"v52_labels has no curve entry for {cid}")
        node_order = curve_labels.node_order
        target_node = target_endpoint_node(node_order, rec.intent.direction)
        target_endpoint = "B" if rec.intent.direction == "forward" else "A"

        cache_key = (cid, int(rec.node_index), int(target_node))
        d_end = d_end_cache.get(cache_key)
        if d_end is None:
            keys = np.asarray(
                [int(k) for k in rec.intent.key_rows()], dtype=np.int64)
            mobile = np.asarray(
                dataset.frame_xyz(cid, rec.node_index),
                dtype=np.float64)[keys]
            target = np.asarray(
                dataset.frame_xyz(cid, target_node),
                dtype=np.float64)[keys]
            d_end = kabsch_rmsd_v52(mobile, target)
            d_end_cache[cache_key] = d_end

        degenerate = (
            cid in DEGENERATE_TS_CURVES
            and int(rec.node_index) == int(dataset.curve_ts_index(cid)))
        _require(rec.label_status in MOVE_STATUS_MAP, "MOVE_STATUS_ENUM",
                 f"{rec.sample_id}: label_status {rec.label_status!r} has no "
                 "move_status mapping")
        lab = EndpointStopSampleLabel(
            sample_id=rec.sample_id,
            curve_id=cid,
            node_index=int(rec.node_index),
            intent_id=rec.intent.intent_id,
            direction=rec.intent.direction,
            target_endpoint=target_endpoint,
            target_endpoint_node=int(target_node),
            d_end_angstrom=float(d_end),
            endpoint_stop_label=endpoint_stop_label_from_distance(
                d_end,
                positive_window=positive_window,
                negative_window=negative_window),
            degenerate_flag=bool(degenerate),
            move_status=MOVE_STATUS_MAP[rec.label_status])
        labels[rec.sample_id] = lab
        curve_rows.setdefault(cid, []).append(lab)

    import statistics

    summaries: dict[str, CurveEndpointStopSummary] = {}
    for cid, rows in curve_rows.items():
        d_ends = sorted(lab.d_end_angstrom for lab in rows)
        node_order = v52_labels.curves[cid].node_order
        summaries[cid] = CurveEndpointStopSummary(
            curve_id=cid,
            endpoint_a_node=int(node_order[0]),
            endpoint_b_node=int(node_order[-1]),
            ts_index=int(v52_labels.curves[cid].ts_index),
            n_rows=len(rows),
            n_pos=sum(1 for lab in rows if lab.endpoint_stop_label == 1),
            n_neg=sum(1 for lab in rows if lab.endpoint_stop_label == 0),
            n_mask=sum(1 for lab in rows if lab.endpoint_stop_label == -1),
            n_degenerate=sum(1 for lab in rows if lab.degenerate_flag),
            d_end_min=float(d_ends[0]),
            d_end_median=float(statistics.median(d_ends)),
            d_end_max=float(d_ends[-1]))

    return EndpointStopLabelTable(
        by_sample=labels,
        curves=summaries,
        positive_window=float(positive_window),
        negative_window=float(negative_window))


# ---------------------------------------------------------------------------
# V5.3-MP `endpoint_label_spec_v1` — executable Endpoint Completion spec
# (V5.3-MP plan §4.3 positive/negative/Mask rules + §4.5 coverage checks;
# MoveProcessContract v1 §5.3 + §10.1).  numpy-only; no torch.
# ---------------------------------------------------------------------------

#: Label spec id (plan §4.3; run-record ``label_spec``).
ENDPOINT_LABEL_SPEC_V1: Final[str] = "endpoint_label_spec_v1"

#: Positive window ``w_pos`` (plan §4.3 候选 0.10 Å): neighborhood frames
#: within ``d_end <= w_pos`` may extend the true endpoint frame to positive.
W_POS_ANGSTROM_V1: Final[float] = 0.10
#: Negative window ``w_neg`` (plan §4.3): ``d_end > w_neg`` ⇒ far from the
#: target endpoint (mid/product-side negatives).  ``w_pos < d_end <= w_neg``
#: is the ambiguous middle that stays Mask.
W_NEG_ANGSTROM_V1: Final[float] = 0.30
#: TS hard-negative window (plan §4.3 负例 2: ``d_key_TS <= 0.04 Å`` 永远负例).
TS_HARD_NEGATIVE_WINDOW_ANGSTROM_V1: Final[float] = 0.04
#: Tolerance for bond-edit completion (plan §4.3 正例 2: 目标键编辑完成 —
#: all required bond distances within this tolerance of the target frame).
BOND_EDIT_TOLERANCE_ANGSTROM_V1: Final[float] = 0.30
#: Minimum legal pairwise interatomic distance (geometry legality /
#: collision gate; plan §4.3 Mask 2 碰撞).
MIN_INTERATOMIC_ANGSTROM_V1: Final[float] = 0.65
#: Energy window for the (diagnostic-only, v1) conformer flag
#: (plan §4.3 正例 3 / 负例-构象: GFN-2 相对参考端点 ≤ 5 kcal/mol).
ENERGY_WINDOW_KCAL_MOL_V1: Final[float] = 5.0
#: Hartree → kcal/mol (CODATA; used only by the diagnostic energy flag).
HARTREE_TO_KCAL_MOL_V1: Final[float] = 627.5095

#: Three-way label values (plan §4.3): +1 positive / 0 negative / −1 Mask.
LABEL_POSITIVE_V1: Final[int] = 1
LABEL_NEGATIVE_V1: Final[int] = 0
LABEL_MASK_V1: Final[int] = -1

#: Label statuses that MUST be loaded (mirrors the V5.2-R refusal's intent).
FULL_LABEL_STATUSES_V1: Final[tuple[str, ...]] = (
    "ok", "terminal", "masked", "conflict")


class EndpointLabelSpecV1Error(RuntimeError):
    """Typed refusal of the ``endpoint_label_spec_v1`` builder."""

    def __init__(self, code: str, detail: str = "") -> None:
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _require_v1(cond: bool, code: str, detail: str) -> None:
    if not cond:
        raise EndpointLabelSpecV1Error(code, detail)


def signed_edit_pairs_v1(
    intent,
) -> tuple[tuple[int, int, float, float, float], ...]:
    """Every signed bond edit ``(i, j, B_src, B_tgt, ΔB)`` of the intent.

    Derived ONLY from the intent's endpoint graphs (``pair_orders``);
    pairs with ``ΔB == 0`` (spectator bonds) are not edits and are
    excluded.  ``i < j`` always.  ΔB classes follow the G1/change
    vocabulary: *forming* (bonded only in G_tgt), *breaking* (bonded only
    in G_src), *order-changed* (bonded at both ends with different order).
    """
    out: list[tuple[int, int, float, float, float]] = []
    for (i, j), (b_src, b_tgt) in sorted(intent.pair_orders().items()):
        delta = float(b_tgt) - float(b_src)
        if delta == 0.0:
            continue
        out.append((int(i), int(j), float(b_src), float(b_tgt), delta))
    return tuple(out)


def _bond_edit_complete_pairs(
    pairs: Sequence[tuple[int, int, float, float, float]],
    cur: np.ndarray,
    tgt: np.ndarray,
    tol: float,
) -> bool:
    """The bond-edit completion check over PRECOMPUTED edit pairs (the
    per-row hot path; identical semantics to :func:`bond_edit_complete_v1`).
    """
    for i, j, _b_src, _b_tgt, delta in pairs:
        d_cur = float(np.linalg.norm(cur[i] - cur[j]))
        d_tgt = float(np.linalg.norm(tgt[i] - tgt[j]))
        if delta > 0.0 and _b_src == 0.0:          # forming
            if not (d_cur <= d_tgt + tol):
                return False
        elif delta < 0.0 and _b_tgt == 0.0:        # breaking
            if not (d_cur >= d_tgt - tol):
                return False
        else:                                      # order-changed
            if not (abs(d_cur - d_tgt) <= tol):
                return False
    return True


def bond_edit_complete_v1(
    intent,
    cur_xyz,
    tgt_xyz,
    *,
    tolerance: float = BOND_EDIT_TOLERANCE_ANGSTROM_V1,
) -> bool:
    """Plan §4.3 目标键编辑完成 — label-side completion check.

    For every signed edit ``(i, j, ΔB)`` of the INTENT, compare the
    current-frame distance ``d_ij(cur)`` with the distance in the intent's
    TARGET endpoint frame ``d_ij(tgt)``:

    * forming (ΔB > 0, absent in G_src): ``d_cur <= d_tgt + tol``;
    * breaking (ΔB < 0, absent in G_tgt): ``d_cur >= d_tgt - tol``;
    * order-changed (present in both graphs, different order):
      ``|d_cur - d_tgt| <= tol``.

    All edits must pass.  **Input-legality note**: using the reference
    endpoint frame here is LEGAL because labels are supervision, not
    model input — the FORBIDDEN-input discipline of MoveProcessContract
    §3.1 (no reference endpoint XYZ) applies to ``model.forward``, not to
    offline label construction.  The trained model never sees the
    reference frame; it only learns from the resulting labels.
    """
    cur = np.asarray(cur_xyz, dtype=np.float64).reshape(-1, 3)
    tgt = np.asarray(tgt_xyz, dtype=np.float64).reshape(-1, 3)
    if cur.shape != tgt.shape or cur.shape[0] == 0:
        raise EndpointLabelSpecV1Error(
            "BOND_EDIT_SHAPE", f"current {cur.shape} vs target {tgt.shape}")
    return _bond_edit_complete_pairs(
        signed_edit_pairs_v1(intent), cur, tgt, float(tolerance))


def geometry_legal_v1(
    xyz,
    *,
    min_interatomic: float = MIN_INTERATOMIC_ANGSTROM_V1,
) -> bool:
    """Plan §4.3 geometry legality: all finite AND no atomic collision.

    Legal iff every coordinate is finite and the minimum pairwise
    interatomic distance over the FULL frame (all atoms, not just K) is
    ``>= min_interatomic`` (default 0.65 Å).
    """
    arr = np.asarray(xyz, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3 or arr.shape[0] == 0:
        return False
    if not bool(np.all(np.isfinite(arr))):
        return False
    if arr.shape[0] == 1:
        return True
    diff = arr[:, None, :] - arr[None, :, :]
    sq = (diff * diff).sum(axis=-1)
    iu = np.triu_indices(arr.shape[0], k=1)
    return bool(np.sqrt(sq[iu]).min() >= float(min_interatomic))


def curve_node_energies_v1(dataset, curve_id: str) -> dict[int, float] | None:
    """Per-node energies (Hartree) of a curve, or None when unavailable.

    The 20k six-table build (plan §4.2/S0-B) adds an ``energy`` column to
    ``irc_node``; the legacy 50-curve delivery ships it empty.  This
    helper reads through the dataset's public/light surfaces only:

    * a dataset exposing ``curve_energies(curve_id)`` (the V5.3-MP
      sharded loader) wins;
    * otherwise the in-memory ``_raw`` irc_node rows of
      ``KeyAtomDatasetV3`` are scanned.

    Returns None when the column is absent, empty, or partially missing —
    the energy flag is DIAGNOSTIC-ONLY in v1 (never changes labels), so
    unavailable data must never fail the build.
    """
    getter = getattr(dataset, "curve_energies", None)
    if callable(getter):
        doc = getter(str(curve_id))
        return dict(doc) if doc else None
    raw = getattr(dataset, "_raw", None)
    if not raw:
        return None
    rows = raw.get("irc_node") or ()
    out: dict[int, float] = {}
    for row in rows:
        if str(row.get("curve_id")) != str(curve_id):
            continue
        cell = (row.get("energy") or "").strip()
        if not cell:
            return None
        try:
            out[int(row["node_index"])] = float(cell)
        except (KeyError, ValueError):
            return None
    return out or None


@dataclass(frozen=True)
class EndpointSampleLabelV1:
    """One sample row's ``endpoint_label_spec_v1`` record.

    ``endpoint_label``: +1 positive (target endpoint completed) / 0
    negative (start, TS, or clearly not at the target endpoint) / −1 Mask
    (ambiguous / illegal / degenerate).  ``d_end_angstrom`` is the Kabsch
    K-geometry RMSD (proper rotation; :func:`kabsch_rmsd_v52`) to the
    TARGET endpoint frame.  ``energy_ev_within_window`` is None when no
    per-node energies are available (diagnostic-only in v1).
    """

    sample_id: str
    curve_id: str
    node_index: int
    intent_id: str
    direction: str                     # "forward" | "reverse"
    fold: int | None
    label_status: str                  # R3 Move-teacher status (raw column)
    #: +1 pos / 0 neg / −1 Mask (plan §4.3)
    endpoint_label: int
    d_end_angstrom: float
    #: Kabsch K-geometry RMSD to the TS frame (Å) — provenance of the
    #: TS hard-negative rule (plan §4.3 负例 2 ``d_key_TS <= 0.04``)
    d_key_ts_angstrom: float
    bond_edit_complete: bool
    geometry_legal: bool
    #: True when the row sits on the curve's trusted TS frame
    ts_row: bool
    #: True when the row sits on the OPPOSITE (start) endpoint
    start_row: bool
    #: True when label_status == "terminal" (no_future_contact)
    terminal_row: bool
    #: True when the whole curve is degenerate (TS ≈ endpoint or listed
    #: in DEGENERATE_TS_CURVES) — every row of such a curve is Mask
    degenerate_curve: bool
    #: |E_node − E_target_endpoint| ≤ 5 kcal/mol; None when unavailable
    energy_ev_within_window: bool | None


@dataclass(frozen=True)
class EndpointLabelTableV1:
    """``endpoint_label_spec_v1`` labels over a FULL-status dataset load.

    ``by_sample`` covers every loaded row (terminal/masked/conflict rows
    included — eligibility independence, contract §5.3).  Ordered,
    deterministic reporting via :meth:`report`.
    """

    by_sample: Mapping[str, EndpointSampleLabelV1]
    positive_window: float
    negative_window: float
    #: curve_id → reaction-family label (report-only; plan §4.5 覆盖率)
    curve_families: Mapping[str, str] | None = None

    @property
    def n_rows(self) -> int:
        return len(self.by_sample)

    def count_label(self, label: int) -> int:
        return sum(1 for lab in self.by_sample.values()
                   if lab.endpoint_label == label)

    @property
    def n_positive(self) -> int:
        return self.count_label(LABEL_POSITIVE_V1)

    @property
    def n_negative(self) -> int:
        return self.count_label(LABEL_NEGATIVE_V1)

    @property
    def n_mask(self) -> int:
        return self.count_label(LABEL_MASK_V1)

    def report(self) -> dict:
        """Plan §4.5 coverage / hard-negative report (deterministic).

        Overall pos/neg/Mask counts; per-curve positives; effective
        positive coverage (fraction of curves with >= 1 positive);
        per-family coverage when ``curve_families`` was supplied;
        per-fold counts; the degenerate-curve list; TS-window vs
        TS-negative counts (equal when no TS-window row is
        degenerate/conflicting); start-row vs start-negative counts.
        Keys are sorted for byte-stable audits.
        """
        labs = [self.by_sample[sid] for sid in sorted(self.by_sample)]
        # single-pass aggregation (O(n_rows)) — the per-group dict
        # comprehensions of the naive form are O(groups × rows) and do not
        # scale to the 20k build (V5.3-MP S0-E lesson, 2026-10-22).
        n_pos = n_neg = n_mask = 0
        n_ts_window = 0
        n_ts_negative = 0
        n_start_rows = 0
        n_start_negative = 0
        n_terminal = 0
        n_terminal_positive = 0
        degenerate_curves: set[str] = set()
        per_curve: dict[str, dict] = {}
        per_fold: dict[str, dict] = {}
        for lab in labs:
            lab_pos = lab.endpoint_label == LABEL_POSITIVE_V1
            lab_neg = lab.endpoint_label == LABEL_NEGATIVE_V1
            lab_mask = lab.endpoint_label == LABEL_MASK_V1
            n_pos += lab_pos
            n_neg += lab_neg
            n_mask += lab_mask
            if lab.d_key_ts_angstrom <= TS_HARD_NEGATIVE_WINDOW_ANGSTROM_V1:
                n_ts_window += 1
                n_ts_negative += bool(lab_neg)
            if lab.start_row:
                n_start_rows += 1
                n_start_negative += bool(lab_neg)
            if lab.terminal_row:
                n_terminal += 1
                n_terminal_positive += bool(lab_pos)
            if lab.degenerate_curve:
                degenerate_curves.add(lab.curve_id)
            c = per_curve.get(lab.curve_id)
            if c is None:
                c = per_curve[lab.curve_id] = {
                    "n_rows": 0, "n_pos": 0, "n_neg": 0, "n_mask": 0,
                    "degenerate": False}
            c["n_rows"] += 1
            c["n_pos"] += bool(lab_pos)
            c["n_neg"] += bool(lab_neg)
            c["n_mask"] += bool(lab_mask)
            c["degenerate"] = c["degenerate"] or bool(lab.degenerate_curve)
            fkey = "none" if lab.fold is None else str(lab.fold)
            f = per_fold.get(fkey)
            if f is None:
                f = per_fold[fkey] = {
                    "n_rows": 0, "n_pos": 0, "n_neg": 0, "n_mask": 0}
            f["n_rows"] += 1
            f["n_pos"] += bool(lab_pos)
            f["n_neg"] += bool(lab_neg)
            f["n_mask"] += bool(lab_mask)

        curves = sorted(per_curve)
        per_curve_positive = {
            cid: per_curve[cid]["n_pos"] for cid in curves}
        n_with_positive = sum(1 for cid in curves
                              if per_curve_positive[cid] >= 1)
        per_fold = {k: per_fold[k] for k in sorted(
            per_fold, key=lambda v: (v == "none", v))}

        per_family: dict[str, dict] | None = None
        if self.curve_families is not None:
            fam_of = {cid: str(self.curve_families.get(cid, "unknown"))
                      for cid in curves}
            fam_agg: dict[str, dict] = {}
            for cid in curves:
                fam = fam_of[cid]
                fa = fam_agg.setdefault(fam, {
                    "n_curves": 0, "n_curves_with_positive": 0,
                    "n_rows": 0, "n_pos": 0, "n_neg": 0, "n_mask": 0})
                fa["n_curves"] += 1
                fa["n_curves_with_positive"] += bool(
                    per_curve_positive[cid] >= 1)
                c = per_curve[cid]
                fa["n_rows"] += c["n_rows"]
                fa["n_pos"] += c["n_pos"]
                fa["n_neg"] += c["n_neg"]
                fa["n_mask"] += c["n_mask"]
            per_family = {}
            for fam in sorted(fam_agg):
                fa = fam_agg[fam]
                fa["effective_positive_coverage"] = (
                    fa["n_curves_with_positive"] / fa["n_curves"]
                    if fa["n_curves"] else 0.0)
                per_family[fam] = fa

        return {
            "spec": ENDPOINT_LABEL_SPEC_V1,
            "positive_window_angstrom": float(self.positive_window),
            "negative_window_angstrom": float(self.negative_window),
            "ts_hard_negative_window_angstrom":
                TS_HARD_NEGATIVE_WINDOW_ANGSTROM_V1,
            "n_rows": len(labs),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_mask": n_mask,
            "n_curves": len(curves),
            "n_curves_with_positive": n_with_positive,
            "effective_positive_coverage": (
                n_with_positive / len(curves) if curves else 0.0),
            "per_curve_positive": per_curve_positive,
            "per_curve": per_curve,
            "degenerate_curves": sorted(degenerate_curves),
            "n_ts_window_rows": n_ts_window,
            "n_ts_negative": n_ts_negative,
            "n_start_rows": n_start_rows,
            "n_start_negative": n_start_negative,
            "n_terminal_rows": n_terminal,
            "n_terminal_positive": n_terminal_positive,
            "per_fold": per_fold,
            "per_family": per_family,
        }

    def to_rows(self) -> list[dict]:
        """Flat row dicts (sorted by sample_id) for audit writers."""
        rows: list[dict] = []
        for sid in sorted(self.by_sample):
            lab = self.by_sample[sid]
            rows.append({
                "sample_id": lab.sample_id,
                "curve_id": lab.curve_id,
                "node_index": lab.node_index,
                "intent_id": lab.intent_id,
                "direction": lab.direction,
                "fold": lab.fold,
                "label_status": lab.label_status,
                "endpoint_label": lab.endpoint_label,
                "d_end_angstrom": lab.d_end_angstrom,
                "d_key_ts_angstrom": lab.d_key_ts_angstrom,
                "bond_edit_complete": lab.bond_edit_complete,
                "geometry_legal": lab.geometry_legal,
                "ts_row": lab.ts_row,
                "start_row": lab.start_row,
                "terminal_row": lab.terminal_row,
                "degenerate_curve": lab.degenerate_curve,
                "energy_ev_within_window": lab.energy_ev_within_window,
            })
        return rows


@dataclass(frozen=True)
class _RowFeaturesV1:
    """Internal per-row features (pass 1) before the precedence fold."""

    sample_id: str
    curve_id: str
    node_index: int
    intent_id: str
    direction: str
    fold: int | None
    label_status: str
    d_end: float
    d_key_ts: float
    bond_edit_complete: bool
    geometry_legal: bool
    energy_flag: bool | None
    ts_row: bool
    start_row: bool
    target_row: bool
    terminal_row: bool

    @property
    def d_key_ts_windowed(self) -> bool:
        return self.d_key_ts <= TS_HARD_NEGATIVE_WINDOW_ANGSTROM_V1


def _endpoint_label_v1(row: _RowFeaturesV1, degenerate: bool) -> int:
    """Frozen precedence fold of plan §4.3 (documented order):

    1. degenerate curve OR mapping conflict (``label_status ==
       "conflict"``) ⇒ **Mask** (wins over everything);
    2. TS-hard-negative: ``d_key_ts <= 0.04 Å`` (TS frame + TS
       neighborhood) ⇒ **negative** (plan §4.3 负例 2: 永远负例);
    3. START-endpoint row (the intent's opposite endpoint) ⇒ **negative**
       (负例 1);
    4. POSITIVE: (a) the TARGET endpoint frame itself, provided geometry
       is legal (endpoint identity is independently verified — contract
       §5.3); (b) neighborhood: ``d_end <= w_pos`` AND bond edits
       complete AND geometry legal; (c) terminal rows are positive ONLY
       under (b)'s conditions (terminal ≠ endpoint, 负例/正例 4);
    5. terminal row that failed the positive test ⇒ **Mask** (Mask 4);
    6. MID-NEGATIVE: ``d_end > w_neg`` ⇒ **negative** — this is exactly
       负例 3 (edits incomplete and far) ∪ 负例 4 (product-side but
       ``d_end > w_neg``);
    7. anything else (middle window with undecidable conformation;
       edits incomplete / collision within ``w_pos``) ⇒ **Mask** (Mask
       1–2).

    A collision (illegal geometry) therefore vetoes near-endpoint
    POSITIVES (step 4) but never rewrites TS/start rows, which stay
    negative by rule; energy is never consulted (diagnostic-only).
    """
    if degenerate or row.label_status == "conflict":
        return LABEL_MASK_V1
    if row.d_key_ts_windowed:
        return LABEL_NEGATIVE_V1
    if row.start_row:
        return LABEL_NEGATIVE_V1
    if row.target_row and row.geometry_legal:
        return LABEL_POSITIVE_V1
    if (row.d_end <= W_POS_ANGSTROM_V1 and row.bond_edit_complete
            and row.geometry_legal):
        return LABEL_POSITIVE_V1
    if row.terminal_row:
        return LABEL_MASK_V1
    if row.d_end > W_NEG_ANGSTROM_V1:
        return LABEL_NEGATIVE_V1
    return LABEL_MASK_V1


def build_endpoint_labels_spec_v1(
    dataset: KeyAtomDatasetV3,
    v52_labels: V52LabelTable,
    *,
    curve_families: Mapping[str, str] | None = None,
) -> EndpointLabelTableV1:
    """Build the ``endpoint_label_spec_v1`` Endpoint Completion table.

    Executable form of plan §4.3 (candidate supervision definition —
    must still pass S0-E hard-negative/coverage checks; it is NOT
    verified chemical endpoint truth, contract §10).  Caller contract:

    * ``dataset`` MUST be loaded with the FULL label status set
      (``ok/terminal/masked/conflict``) — a terminal-less load is refused
      (typed ``TERMINAL_NOT_LOADED``, mirroring
      :func:`build_endpoint_stop_labels`) because terminal rows are
      exactly where the old defect hid;
    * ``v52_labels`` (``dataset.v52_labels()`` output) supplies the
      canonical A→B ``node_order`` per curve and the per-row
      ``d_key_ts`` used by the TS hard-negative rule; it must cover the
      same sample rows.

    Per row: ``K = intent.key_rows()``; ``d_end`` = Kabsch proper-
    rotation RMSD of the current K geometry against the TARGET endpoint
    frame's K geometry (forward → path-end B, reverse → path-start A via
    :func:`target_endpoint_node`); ``bond_edit_complete`` via
    :func:`bond_edit_complete_v1` (label-side use of the reference
    endpoint frame is legal — supervision, not model input);
    ``geometry_legal`` via :func:`geometry_legal_v1`; the
    ``energy_ev_within_window`` flag via :func:`curve_node_energies_v1`
    is recorded but NEVER changes a label in v1 (conformer-positive
    extension stays a candidate).  Degenerate curves (auto: TS-frame
    ``d_end <= w_pos``, union with :data:`DEGENERATE_TS_CURVES`) are
    flagged and fully Masked.  See :func:`_endpoint_label_v1` for the
    frozen precedence order.
    """
    _require_v1(
        "terminal" in dataset.label_statuses, "TERMINAL_NOT_LOADED",
        "dataset was loaded without the 'terminal' status; "
        "endpoint_label_spec_v1 requires the FULL status set "
        "(ok/terminal/masked/conflict) so no_future_contact rows enter "
        "PROCESS data (plan §4.3 Mask 4; contract §5.3 eligibility "
        "independence)")
    missing_statuses = [s for s in FULL_LABEL_STATUSES_V1
                        if s not in dataset.label_statuses]
    _require_v1(not missing_statuses, "STATUS_SET_NOT_LOADED",
                f"dataset label_statuses misses {missing_statuses}; "
                "build with the full ORA-2 #3 set")
    if not (float(W_POS_ANGSTROM_V1) <= float(W_NEG_ANGSTROM_V1)):
        raise EndpointLabelSpecV1Error(
            "ENDPOINT_WINDOW_ORDER",
            f"positive window {W_POS_ANGSTROM_V1} > negative "
            f"{W_NEG_ANGSTROM_V1}")

    loaded_ids = {rec.sample_id for rec in dataset.samples}
    missing = loaded_ids - set(v52_labels.by_sample)
    _require_v1(not missing, "V52_LABEL_COVERAGE",
                f"v52_labels misses {len(missing)} loaded sample rows "
                f"(e.g. {sorted(missing)[:3]}); rebuild via "
                "dataset.v52_labels() with the SAME filters")

    # --- pass 1: per-row features + degenerate-curve auto-detection ---------
    rows: list[_RowFeaturesV1] = []
    d_end_cache: dict[tuple[str, int, int], float] = {}
    target_frames: dict[tuple[str, int], np.ndarray] = {}
    energy_cache: dict[str, dict[int, float] | None] = {}
    degenerate_curves: set[str] = set()
    last_frame_key: tuple[str, int] | None = None
    last_frame: np.ndarray | None = None
    # per-(curve, node) / per-intent caches (all memoize PURE functions of the
    # frozen tables — bit-identical results, O(1) repeat cost; the naive
    # per-row recomputation does not scale to the 20k build)
    geom_cache: dict[tuple[str, int], bool] = {}
    edit_pairs_cache: dict[str, tuple] = {}
    key_rows_cache: dict[str, np.ndarray] = {}

    for rec in dataset.samples:
        cid = rec.curve_id
        curve_labels = v52_labels.curves.get(cid)
        _require_v1(curve_labels is not None, "V52_CURVE_COVERAGE",
                    f"v52_labels has no curve entry for {cid}")
        order = curve_labels.node_order
        direction = rec.intent.direction
        target_node = target_endpoint_node(order, direction)
        start_node = target_endpoint_node(
            order, "reverse" if direction == "forward" else "forward")
        node = int(rec.node_index)

        # current frame (1-slot cache: consecutive rows share curve/node)
        frame_key = (cid, node)
        if last_frame is None or frame_key != last_frame_key:
            last_frame = np.asarray(
                dataset.frame_xyz(cid, node), dtype=np.float64)
            last_frame_key = frame_key
        full_cur = last_frame

        # target endpoint frame (cached per (curve, target node)); the
        # K-restricted copy feeds the Kabsch d_end, the FULL frame feeds
        # the bond-edit completion check
        tgt_key = (cid, int(target_node))
        tgt_full = target_frames.get(tgt_key)
        if tgt_full is None:
            tgt_full = np.asarray(
                dataset.frame_xyz(cid, target_node), dtype=np.float64)
            target_frames[tgt_key] = tgt_full
        iid = rec.intent.intent_id
        keys = key_rows_cache.get(iid)
        if keys is None:
            keys = np.asarray(
                [int(k) for k in rec.intent.key_rows()], dtype=np.int64)
            key_rows_cache[iid] = keys
        tgt = tgt_full[keys]

        cache_key = (cid, node, int(target_node))
        d_end = d_end_cache.get(cache_key)
        if d_end is None:
            d_end = kabsch_rmsd_v52(full_cur[keys], tgt)
            d_end_cache[cache_key] = d_end
        edit_pairs = edit_pairs_cache.get(iid)
        if edit_pairs is None:
            edit_pairs = signed_edit_pairs_v1(rec.intent)
            edit_pairs_cache[iid] = edit_pairs
        edit_ok = _bond_edit_complete_pairs(
            edit_pairs, full_cur, tgt_full, BOND_EDIT_TOLERANCE_ANGSTROM_V1)
        geom_ok = geom_cache.get(frame_key)
        if geom_ok is None:
            geom_ok = geometry_legal_v1(
                full_cur, min_interatomic=MIN_INTERATOMIC_ANGSTROM_V1)
            geom_cache[frame_key] = geom_ok

        if cid not in energy_cache:
            energy_cache[cid] = curve_node_energies_v1(dataset, cid)
        energies = energy_cache[cid]
        energy_flag: bool | None = None
        if energies is not None and node in energies \
                and int(target_node) in energies:
            delta_ev = abs(energies[node] - energies[int(target_node)]) \
                * HARTREE_TO_KCAL_MOL_V1
            energy_flag = bool(delta_ev <= ENERGY_WINDOW_KCAL_MOL_V1)

        ts_node = int(curve_labels.ts_index)
        row = _RowFeaturesV1(
            sample_id=rec.sample_id,
            curve_id=cid,
            node_index=node,
            intent_id=rec.intent.intent_id,
            direction=direction,
            fold=rec.fold,
            label_status=rec.label_status,
            d_end=float(d_end),
            d_key_ts=float(v52_labels.by_sample[rec.sample_id].d_key_ts),
            bond_edit_complete=edit_ok,
            geometry_legal=geom_ok,
            energy_flag=energy_flag,
            ts_row=node == ts_node,
            start_row=node == int(start_node),
            target_row=node == int(target_node),
            terminal_row=rec.label_status == "terminal")
        rows.append(row)
        # auto-detection: TS frame within w_pos of the target endpoint
        if row.ts_row and row.d_end <= W_POS_ANGSTROM_V1:
            degenerate_curves.add(cid)

    degenerate_curves |= {cid for cid in v52_labels.curves
                          if cid in DEGENERATE_TS_CURVES}
    degenerate_curves &= {row.curve_id for row in rows}

    # --- pass 2: precedence fold --------------------------------------------
    labels: dict[str, EndpointSampleLabelV1] = {}
    for row in rows:
        degenerate = row.curve_id in degenerate_curves
        labels[row.sample_id] = EndpointSampleLabelV1(
            sample_id=row.sample_id,
            curve_id=row.curve_id,
            node_index=row.node_index,
            intent_id=row.intent_id,
            direction=row.direction,
            fold=row.fold,
            label_status=row.label_status,
            endpoint_label=_endpoint_label_v1(row, degenerate),
            d_end_angstrom=row.d_end,
            d_key_ts_angstrom=row.d_key_ts,
            bond_edit_complete=row.bond_edit_complete,
            geometry_legal=row.geometry_legal,
            ts_row=row.ts_row,
            start_row=row.start_row,
            terminal_row=row.terminal_row,
            degenerate_curve=degenerate,
            energy_ev_within_window=row.energy_flag)

    return EndpointLabelTableV1(
        by_sample=labels,
        positive_window=float(W_POS_ANGSTROM_V1),
        negative_window=float(W_NEG_ANGSTROM_V1),
        curve_families=(dict(curve_families)
                        if curve_families is not None else None))


__all__ = [
    "SCHEMA",
    "ENDPOINT_STOP_POSITIVE_WINDOW_ANGSTROM",
    "ENDPOINT_STOP_NEGATIVE_WINDOW_ANGSTROM",
    "DEGENERATE_TS_CURVES",
    "MOVE_STATUS_MAP",
    "EndpointStopLabelError",
    "EndpointStopSampleLabel",
    "CurveEndpointStopSummary",
    "EndpointStopLabelTable",
    "endpoint_stop_label_from_distance",
    "target_endpoint_node",
    "build_endpoint_stop_labels",
    # V5.3-MP endpoint_label_spec_v1 (plan §4.3)
    "ENDPOINT_LABEL_SPEC_V1",
    "W_POS_ANGSTROM_V1",
    "W_NEG_ANGSTROM_V1",
    "TS_HARD_NEGATIVE_WINDOW_ANGSTROM_V1",
    "BOND_EDIT_TOLERANCE_ANGSTROM_V1",
    "MIN_INTERATOMIC_ANGSTROM_V1",
    "ENERGY_WINDOW_KCAL_MOL_V1",
    "HARTREE_TO_KCAL_MOL_V1",
    "LABEL_POSITIVE_V1",
    "LABEL_NEGATIVE_V1",
    "LABEL_MASK_V1",
    "FULL_LABEL_STATUSES_V1",
    "EndpointLabelSpecV1Error",
    "EndpointSampleLabelV1",
    "EndpointLabelTableV1",
    "signed_edit_pairs_v1",
    "bond_edit_complete_v1",
    "geometry_legal_v1",
    "curve_node_energies_v1",
    "build_endpoint_labels_spec_v1",
]
