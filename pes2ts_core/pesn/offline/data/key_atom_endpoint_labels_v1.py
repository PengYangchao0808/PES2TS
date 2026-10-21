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
]
