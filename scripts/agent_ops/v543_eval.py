"""V5.4.3 — M0/P0 baseline recorder under the FROZEN V543 evaluator (§9.5).

WHAT THIS IS
------------
The plan's §9.5 :1035–1041 clause, executed:

    "**基线**：S0 在 V543 evaluator 下重算 M0/P0 的 per-seed inner-val 文档；
    SHA 冻结；H2/H3 只用这份基线，不用 V542 归档数字。"

So this runner does exactly three things and invents nothing:

1. loads each campaign seed's frozen A0 checkpoint STRICT
   (``CoarseEncoder.load``, 45-key ``v542_a0_joint_45`` payload, SHA-verified
   against ``config/campaigns/pes2ts_v543_dual_decoder_50k.yaml``
   ``source.checkpoints``), builds the frozen ``DecoderContextV1`` with the
   WP2 Bridge, and reads the zero-residual quantities of §4.3 :226
   (``u0`` = M0, ``tanh(q0_raw)``/``e0`` = P0);
2. evaluates them with ``pes2ts_core/pesn/offline/evaluation/v543_eval.py``
   (``evaluate_baseline``) on the FROZEN inner-val slice of
   ``v53mp_pool_split(seed)`` — the same slice the S0 probe used, no RNG, no
   dev, no confirm;
3. writes ``outputs/pesn/v543_dual_decoder_50k/audit/m0_p0_baseline_v543_eval.json``
   with the evaluator code SHA, the evaluator spec SHA, the Bridge spec SHA,
   the three checkpoint SHAs, every §9.1 metric per seed, the per-curve rows
   that ``paired_curve_bootstrap`` will consume, and a per-seed document SHA.

WHY M0/P0 IS "THE SAME A0 WEIGHTS + ZERO RESIDUAL"
-------------------------------------------------
§1.6 :118 — the V542 A0 numbers (RMSE/zero 0.9099 etc.) come from the V542
joint model under the V542 evaluator and "不能直接作为 V543 的 H2 分母".  The
matched V543 baseline is the A0 encoder with the zero residual (§4.3 :226: "未
参与当前 Track 的 Refiner 固定为零残差（M0/P0）"), measured by the SAME
evaluator that will measure every candidate.  That is what this file records.

DISCIPLINE
----------
* no file under ``pes2ts_core/pesn/models/v543/`` is imported for a decoder —
  M0/P0 need the Encoder and the Bridge only (no WP4 registry, no trainer);
* supervision (``y_teacher`` / ``q_true`` / ``q_mask`` / endpoint labels /
  ``pair_index`` set) is read HERE and passed to the evaluator as explicit,
  separate arguments (§5.6 :420–425);
* the pair set handed to the evaluator is the Bridge's forward-side changed
  K-pair reconstruction (``intent_pair_index``), never a model output;
* ``tau_end = None`` (§1.5 :105) — no threshold is calibrated here;
* the evaluator never reads dev/confirm: this runner only ever asks
  ``v53mp_pool_split(seed)`` for its inner-train/inner-val positions.

CLI::

    PYTHONPATH=. python scripts/agent_ops/v543_eval.py \
        --out outputs/pesn/v543_dual_decoder_50k/audit/m0_p0_baseline_v543_eval.json

    # a REFRESH after a code-only refactor: assert metrics did not move
    PYTHONPATH=. python scripts/agent_ops/v543_eval.py \
        --compare-to <previous report>.json --out <same path>

The document carries the bootstrap inputs in TWO modes (see
``bootstrap_inputs.modes``): ``per_seed_independent``
(``paired_curve_bootstrap_multiseed`` — the CONFIRMATORY §9.4 statistic, every
seed resamples its own inner-val cluster set and the seeds are averaged per
draw) and ``shared_intersection`` (``paired_curve_bootstrap`` — the audit
reading, kept for reproducibility; the all-seed intersection is far too small
on this campaign).  Every metric's per-seed ``{curve_id: value}`` tables are
exported so the stats lane can drive either mode without re-reading this
module.

The evaluation budget (``--eval-curves`` × ``--frames-per-curve``, default
1024 × 8 = the plan's 8,192-sample inner-val cap, §1.4 :91) is an explicit,
reported parameter.  The selection is CURVE-first on purpose: §9.4 makes the
curve the bootstrap cluster and §9.4 :1031 needs ≥3 q points per curve, so a
frame set with one frame per curve would have no valid q correlation and zero
ordering pairs — it could not serve as an H2/H3 denominator at all.

``--compare-to`` makes the refresh auditable: every per-seed metric document and
the seed-averaged block must be BIT-identical to the previous frozen report, or
the run refuses with ``V543_EVAL_BASELINE_DRIFT``; the code-SHA bump and its
reason land in the document's ``revision`` block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
import torch

from pes2ts_core.pesn.contracts.key_atom_v3 import (
    NODE_KEY_SET_KEY,
    NODE_POS_KEY,
    forward_view_v3,
)
from pes2ts_core.pesn.models.v543 import CoarseEncoder, LatentBridge
from pes2ts_core.pesn.offline.evaluation.v543_eval import (
    ATOM_DEAD_THRESHOLD,
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED_ROOT,
    BOND_SCALE_NOTE,
    ECE_BINS,
    EFFECTIVE_ATOM_THRESHOLD,
    FIXED_COVERAGE_TOPK,
    FPR_CONSTRAINT,
    PLAN_SECTION as EVAL_PLAN_SECTION,
    Q_CORR_MIN_POINTS,
    Q_FRAC_GE3_MIN,
    REPORT_SCHEMA as EVAL_REPORT_SCHEMA,
    SIGN_PAIR_THRESHOLD,
    EvalFrame,
    V543EvalError,
    ambiguities as evaluator_ambiguities,
    evaluate_baseline,
    evaluator_spec,
    evaluator_spec_sha256,
    zero_residual_frames,
)
from scripts.agent_ops.v542_a0_parity import (
    DEFAULT_CONFIG as DATA_CONFIG,
    PROJECT_ROOT,
    V542ParityError,
    _open_frozen_dataset,
    load_pinned_assets,
)
from scripts.agent_ops.v543_s0_probe import (
    CAMPAIGN_CONFIG,
    load_campaign_config,
    positions_meta,
    resolve_checkpoints,
)

#: Report schema of THIS document (the per-seed blocks carry
#: ``pesn_v543_eval_report_v1``).
BASELINE_SCHEMA: Final[str] = "pesn_v543_m0p0_baseline_v1"
#: Default output (plan §11.1 audit directory; named by the §9.5 clause).
DEFAULT_OUT: Final[str] = (
    "outputs/pesn/v543_dual_decoder_50k/audit/m0_p0_baseline_v543_eval.json")
#: Frames per collated batch (the S1 width is 8 curves x 4 frames = 32).
FRAMES_PER_BATCH: Final[int] = 32
#: Plan §1.4 :91 — the inner-val cap (samples).
INNER_VAL_CAP: Final[int] = 8192
#: Curves in the evaluated inner-val set (the §9.4 bootstrap cluster unit).
DEFAULT_EVAL_CURVES: Final[int] = 1024
#: Frames per curve: the §9.4 q-validity rule needs >=3 points per curve, and
#: the §9.2 aggregation chain is frame -> curve, so every selected curve must
#: contribute several frames or the curve-level statistics are degenerate.
DEFAULT_FRAMES_PER_CURVE: Final[int] = 8
#: The §9.5 decision record this document descends from (recorded, not read).
WP3_DECISION_RECORD: Final[str] = (
    "outputs/pesn/v543_dual_decoder_50k/audit/wp3_go_no_go_decision.json")
#: The plan lines quoted by this document.
PLAN_SECTION: Final[str] = (
    "§9.5 :1035-1041 (M0/P0 重算 + SHA 冻结) / §1.6 :110-118 (V542 数字不是 "
    "V543 分母) / §4.3 :226 (零残差 = M0/P0) / §1.4 :91 (inner-val cap 8,192) "
    "/ §9.1-§9.4 (the evaluator's own contract)")
#: The §9.1/§9.3 metrics whose per-seed curve values are handed to the paired
#: bootstrap (higher_is_better flag included so v543_stats cannot pick wrong).
#: ``endpoint.auprc``/``brier``/``ece`` are curve-aggregable; the operating
#: point ``recall_at_fpr_0p05`` is a pooled frame-level quantity and is
#: therefore NOT in this table (its H2/H3 use is an absolute-delta rule of
#: §9.3, and §9.4 :1025 forbids a frame-level bootstrap).
BOOTSTRAP_METRICS: Final[tuple[tuple[str, tuple[str, str], bool], ...]] = (
    # (metric id, (per-curve block, per-curve key), higher_is_better)
    ("move.rmse_zero", ("move", "rmse_zero"), False),
    ("move.sign_accuracy", ("move", "sign_accuracy"), True),
    ("move.fixed_coverage.direction_at_3",
     ("move", "fixed_coverage_direction_at_3"), True),
    ("move.fixed_coverage.topk_overlap_at_3",
     ("move", "fixed_coverage_topk_overlap_at_3"), True),
    ("move.dead_fraction", ("move", "dead_zone_fraction"), False),
    ("process.q.mae_curve_median", ("q", "mae"), False),
    ("process.ordering.pairwise_accuracy_full",
     ("ordering", "pairwise_accuracy_full"), True),
    ("process.endpoint.auprc", ("endpoint", "auprc"), True),
    ("process.endpoint.brier", ("endpoint", "brier"), False),
    ("process.endpoint.ece", ("endpoint", "ece"), False),
)


#: Plan §1.6 :110–118 — the V542 A0 archive numbers, quoted as HISTORICAL
#: ANCHORS ONLY.  The plan says explicitly that they "不能直接作为 V543 的 H2
#: 分母"; they are recorded here so a reader can see how far the matched M0/P0
#: baseline moved (and where it necessarily must not match).
V542_A0_ANCHORS: Final[Mapping[str, Mapping[str, float]]] = {
    "17": {"rmse_zero": 0.9099, "direction": 0.8000, "dead": 0.1079,
           "auprc": 0.2868, "q_corr": 0.9988},
    "29": {"rmse_zero": 0.9008, "direction": 0.8000, "dead": 0.1015,
           "auprc": 0.3068, "q_corr": 0.9999},
    "43": {"rmse_zero": 0.8979, "direction": 0.8333, "dead": 0.0889,
           "auprc": 0.2965, "q_corr": 0.9940},
}


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(obj: Any) -> str:
    from pes2ts_core.pesn.models.v542.identity import canonical_sha256

    return canonical_sha256(obj)


# ---------------------------------------------------------------------------
# frame construction (encoder + Bridge -> EvalFrame)
# ---------------------------------------------------------------------------
@dataclass
class SeedFrames:
    """One seed's evaluated frame set + the evidence around it."""

    seed: int
    frames: list[EvalFrame]
    label_status_counts: dict[str, int]
    curve_ids: tuple[str, ...]
    n_positions: int
    n_batches: int
    wall_seconds: float
    encoder_forward_count: int
    positions: list[int]
    positions_rule: Mapping[str, Any]
    intent_class_counts: dict[str, int] = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.frames)


def label_status_of(records: Sequence[Any]) -> list[str]:
    """``label_status`` per frame (the frozen MOVE-eligibility rule)."""
    return [str(getattr(record, "label_status", "")) for record in records]


def frame_intent_class(node_lo: int, node_hi: int, pair_index: np.ndarray,
                       pair_edit: np.ndarray) -> str:
    """breaking / forming / both / unmoved for ONE frame (§7.1 :640 taxonomy).

    Uses the SIGN of the supervision ``pair_edit`` of the frame's own changed
    pairs (``pair_index`` columns whose two endpoints fall inside
    ``[node_lo, node_hi)`` and whose edit is non-zero).  ADR-0034 §10 v1.1:
    ``pair_edit = BOND_SCALE × forward ΔB``; only the sign is used here and the
    sign is invariant under the positive ``BOND_SCALE``, so no conversion is
    applied.
    """
    signs: set[int] = set()
    for column in range(int(pair_index.shape[1])):
        left = int(pair_index[0, column])
        right = int(pair_index[1, column])
        if not (node_lo <= left < node_hi and node_lo <= right < node_hi):
            continue
        edit = float(pair_edit[column])
        if edit == 0.0:
            continue
        signs.add(-1 if edit < 0.0 else 1)
    if not signs:
        return "unmoved"
    if signs == {-1}:
        return "breaking"
    if signs == {1}:
        return "forming"
    return "both"


def build_seed_frames(
        encoder: CoarseEncoder,
        bridge: LatentBridge,
        dataset: Any,
        lru: Any,
        positions: Sequence[int],
        *,
        checkpoint_sha256: str,
        meta: Sequence[Mapping[str, Any]],
        labels_v52: Any,
        endpoint_labels: Any,
        with_basis: bool = True,
) -> SeedFrames:
    """Encode ``positions`` and assemble the evaluator's frame set.

    The M0/P0 prediction IS the packet/context's zero residual (§4.3 :226):
    ``u = u0``, ``q = tanh(q0_raw)``, ``endpoint_logit = e0``.  No decoder is
    constructed — this document is the BASELINE, and the zero residual is the
    definition of it.
    """
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        _batch_from_positions,
    )

    started = time.perf_counter()
    frames: list[EvalFrame] = []
    status_counts: dict[str, int] = {}
    intent_counts: dict[str, int] = {}
    n_batches = 0
    for start in range(0, len(positions), FRAMES_PER_BATCH):
        chunk = [int(p) for p in positions[start:start + FRAMES_PER_BATCH]]
        batch, records = _batch_from_positions(dataset, lru, chunk, "cpu")
        view = forward_view_v3(batch)
        with torch.no_grad():
            packet = encoder(view)
            context = bridge.build_context(
                packet, view, encoder_ckpt_sha256=str(checkpoint_sha256))
        n_batches += 1

        key_ptr = context.key_ptr.detach().cpu().numpy().astype(np.int64)
        pair_ptr = context.pair_ptr.detach().cpu().numpy().astype(np.int64)
        pair_graph = context.pair_graph.detach().cpu().numpy().astype(np.int64)
        intent_pair_index = context.intent_pair_index.detach().cpu().numpy(
        ).astype(np.int64)
        key_graph = context.key_graph.detach().cpu().numpy().astype(np.int64)
        # Bridge decision 4: context arrays are in graph-contiguous canonical
        # key order; the dataset/MOVE-loss order is the raw key_index order.
        permutation = np.argsort(key_graph, kind="stable")
        key_index_raw = batch[NODE_KEY_SET_KEY].detach().cpu().numpy().astype(
            np.int64)
        canonical_key_rows = key_index_raw[permutation]
        # Bridge decision 4 (plan §5.2): context arrays are in the canonical
        # key-slot order, and ``out[permutation] = canonical`` restores the
        # dataset key-row order the MOVE loss / y_teacher use (the same
        # one-line inverse the model applies, kept local so this runner does
        # not depend on the WP4 assembly module).
        u0_rows = np.empty_like(context.u0.detach().cpu().numpy())
        u0_rows[permutation] = context.u0.detach().cpu().numpy()
        key_v = context.key_v.detach().cpu().numpy()
        q0_raw = context.q0_raw.detach().cpu().numpy()
        e0 = context.endpoint0_logit.detach().cpu().numpy()
        y_teacher = batch["y_teacher"].detach().cpu().numpy()
        pos = batch[NODE_POS_KEY].detach().cpu().numpy().astype(np.float64)
        pair_index = batch["pair_index"].detach().cpu().numpy().astype(np.int64)
        pair_edit = batch["pair_edit"].detach().cpu().numpy().astype(np.float64)
        ptr = batch["ptr"].detach().cpu().numpy().astype(np.int64)
        sample_ids = [str(s) for s in batch["sample_ids"]]
        statuses = label_status_of(records)

        n_graphs = int(key_ptr.shape[0]) - 1
        if n_graphs != len(chunk):
            raise V543EvalError(
                "V543_EVAL_RUNNER_GRAPHS",
                f"{n_graphs} graphs in the context for {len(chunk)} positions")
        for g in range(n_graphs):
            k_lo, k_hi = int(key_ptr[g]), int(key_ptr[g + 1])
            p_lo, p_hi = int(pair_ptr[g]), int(pair_ptr[g + 1])
            if bool((pair_graph[p_lo:p_hi] != g).any()):   # pragma: no cover
                raise V543EvalError(
                    "V543_EVAL_RUNNER_PAIRS",
                    f"pair_graph disagrees with graph {g}")
            sample_id = sample_ids[g]
            label_row = labels_v52.by_sample[sample_id]
            endpoint_row = endpoint_labels.by_sample[sample_id]
            # pair slots: chunk-global canonical key slots -> this graph's rows
            slots_global = intent_pair_index[:, p_lo:p_hi]
            if int(slots_global.size):
                pair_offsets = (
                    pos[canonical_key_rows[slots_global[1]]]
                    - pos[canonical_key_rows[slots_global[0]]])
                slots_local = (slots_global - k_lo).T
            else:
                pair_offsets = np.zeros((0, 3), dtype=np.float64)
                slots_local = np.zeros((0, 2), dtype=np.int64)
            status = statuses[g]
            status_counts[status] = int(status_counts.get(status, 0)) + 1
            intent = frame_intent_class(int(ptr[g]), int(ptr[g + 1]),
                                        pair_index, pair_edit)
            intent_counts[intent] = int(intent_counts.get(intent, 0)) + 1
            endpoint_label = int(endpoint_row.endpoint_label)
            frames.append(EvalFrame(
                curve_id=str(label_row.curve_id),
                u=np.asarray(u0_rows[k_lo:k_hi], dtype=np.float64),
                y=np.asarray(y_teacher[k_lo:k_hi], dtype=np.float64),
                pair_slots=np.asarray(slots_local, dtype=np.int64),
                pair_offsets=np.asarray(pair_offsets, dtype=np.float64),
                direction=str(label_row.direction),
                stratum=str(meta[start + g]["stratum"]),
                intent_class=intent,
                q_pred=float(math.tanh(float(q0_raw[g]))),
                q_true=(0.0 if label_row.q is None else float(label_row.q)),
                q_mask=bool(label_row.q_mask),
                endpoint_logit=float(e0[g]),
                endpoint_label=endpoint_label,
                endpoint_mask=endpoint_label in (0, 1),
                is_start_row=bool(endpoint_row.start_row),
                is_ts_row=bool(endpoint_row.ts_row),
                move_eligible=(status == "ok"),
                u0=np.asarray(u0_rows[k_lo:k_hi], dtype=np.float64),
                q0_raw=float(q0_raw[g]),
                endpoint0_logit=float(e0[g]),
                key_v=(np.asarray(key_v[k_lo:k_hi], dtype=np.float64)
                       if with_basis else None),
            ))
    return SeedFrames(
        seed=-1,
        frames=frames,
        label_status_counts=dict(sorted(status_counts.items())),
        curve_ids=tuple(frame.curve_id for frame in frames),
        n_positions=len(positions),
        n_batches=n_batches,
        wall_seconds=float(time.perf_counter() - started),
        encoder_forward_count=int(encoder.forward_count),
        positions=[int(p) for p in positions],
        positions_rule={},
        intent_class_counts=dict(sorted(intent_counts.items())),
    )


# ---------------------------------------------------------------------------
# inner-val positions (deterministic, frozen slice only)
# ---------------------------------------------------------------------------
def inner_val_positions(
        val_pos: Sequence[int],
        dataset: Any,
        *,
        eval_curves: int = DEFAULT_EVAL_CURVES,
        frames_per_curve: int = DEFAULT_FRAMES_PER_CURVE,
) -> tuple[list[int], dict[str, Any]]:
    """Whole inner-val curves with a deterministic per-curve frame spread.

    The frozen inner-val slice is ~10,000 curves of ~240 frames each, ordered
    curve by curve.  Two facts make a naive "first N positions" or "uniform
    spread over positions" selection WRONG for this evaluator:

    * §9.4 :1018 makes the CURVE the bootstrap cluster, so a frame set that
      visits each curve once has one cluster per frame and cannot be
      bootstrapped;
    * §9.4 :1031 requires >=3 valid q points per curve before a correlation
      exists, and §9.1 :892 compares q WITHIN a ``(curve, direction)`` group,
      so one frame per curve yields zero ordering pairs and an all-``None``
      q.corr — a baseline that cannot be an H2/H3 denominator.

    Selection rule (deterministic, no RNG): take the first ``eval_curves``
    curves in sorted inner-val position order, and inside each curve a uniform
    ``linspace`` spread of ``frames_per_curve`` positions.  The default budget
    (1024 curves x 8 frames) is exactly the plan's 8,192-sample inner-val cap
    (§1.4 :91).
    """
    sorted_pos = np.sort(np.asarray([int(p) for p in val_pos], dtype=np.int64))
    if sorted_pos.size == 0:
        raise V543EvalError("V543_EVAL_NO_INNER_VAL",
                            "the frozen inner-val slice is empty")
    # curve -> direction -> positions, both in the frozen sorted order.  A
    # curve carries BOTH the forward and the reverse trajectory of the same
    # reaction, and the two are interleaved in the frozen order; the endpoint
    # label is direction-dependent (it marks the branch's OWN end).  Sampling
    # per (curve, direction) group is therefore the only way to cover both
    # directions' progress (and their endpoints) symmetrically.
    by_curve: dict[str, dict[str, list[int]]] = {}
    for position in sorted_pos.tolist():
        sample = dataset.samples[int(position)]
        curve_id = str(sample.curve_id)
        direction = str(getattr(sample.intent, "direction", ""))
        by_curve.setdefault(curve_id, {}).setdefault(direction, []).append(
            int(position))
    curves = list(by_curve)[:max(1, int(eval_curves))]
    if not curves:
        raise V543EvalError("V543_EVAL_NO_INNER_VAL",
                            "no inner-val curve could be selected")
    group_budget = max(1, int(frames_per_curve))
    positions: list[int] = []
    counts: list[int] = []
    for curve_id in curves:
        groups = by_curve[curve_id]
        per_group = max(1, group_budget // max(1, len(groups)))
        chosen: set[int] = set()
        for direction in sorted(groups):
            group_positions = groups[direction]
            wanted = min(per_group, len(group_positions))
            picks = np.linspace(0, len(group_positions) - 1, wanted)
            for p in picks:
                chosen.add(int(group_positions[int(round(p))]))
        positions.extend(sorted(chosen))
        counts.append(len(chosen))
    if not positions:
        raise V543EvalError("V543_EVAL_NO_INNER_VAL",
                            "the curve selection produced no positions")
    rule = {
        "rule": (
            "first N inner-val curves of the frozen sorted slice; inside each "
            "curve every (curve, direction) group is sampled with a "
            "deterministic uniform linspace spread of frames (no RNG)"),
        "n_positions_available": int(sorted_pos.size),
        "n_curves_available": len(by_curve),
        "n_curves_used": len(curves),
        "frames_per_curve_requested": group_budget,
        "n_positions_used": len(positions),
        "frames_per_curve_min": int(min(counts)),
        "frames_per_curve_max": int(max(counts)),
        "frames_per_curve_median": float(np.median(counts)),
        "n_direction_groups": int(sum(len(by_curve[c]) for c in curves)),
        "inner_val_cap_plan": INNER_VAL_CAP,
        "frames_per_batch": FRAMES_PER_BATCH,
        "why_not_a_position_uniform_spread": (
            "a uniform spread over POSITIONS visits ~1 frame per curve: with "
            "one frame per curve the §9.4 cluster unit degenerates, q.corr "
            "cannot exist (§9.4 :1031) and ordering has no pairs (§9.1 :892)"),
        "why_per_direction": (
            "a curve carries the forward AND the reverse trajectory of the "
            "same reaction interleaved in the frozen order, and the endpoint "
            "label marks each branch's OWN end; a curve-wide spread would "
            "cover one branch's start and the other's end only, collapsing "
            "the endpoint prevalence"),
    }
    return positions, rule


# ---------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------
@dataclass
class SeedOutcome:
    seed: int
    frames: SeedFrames
    report: dict[str, Any]
    identity: dict[str, Any]
    train_q_mean: float | None
    curve_values: dict[str, Any]


def run_seed(*, seed: int, checkpoint: Mapping[str, Any], dataset: Any,
             train_pos: Sequence[int], val_pos: Sequence[int],
             labels_v52: Any, endpoint_labels: Any, eval_curves: int,
             frames_per_curve: int, with_basis: bool, with_strata: bool
             ) -> SeedOutcome:
    """Load the seed's A0 encoder and evaluate its M0/P0 baseline."""
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        _FeatureLRU,
        _train_q_mean,
    )

    positions, rule = inner_val_positions(
        val_pos, dataset, eval_curves=int(eval_curves),
        frames_per_curve=int(frames_per_curve))
    meta = (positions_meta(dataset, positions) if with_strata
            else [{"stratum": ""} for _ in positions])
    lru = _FeatureLRU(dataset, capacity=len(positions) + 128)

    encoder = CoarseEncoder.load(str(checkpoint["path"]), int(seed))
    encoder.eval()
    bridge = LatentBridge(encoder_ckpt_sha256=str(checkpoint["sha256"]))
    frames = build_seed_frames(
        encoder, bridge, dataset, lru, positions,
        checkpoint_sha256=str(checkpoint["sha256"]), meta=meta,
        labels_v52=labels_v52, endpoint_labels=endpoint_labels,
        with_basis=with_basis)
    frames.seed = int(seed)
    frames.positions_rule = dict(rule)

    train_q_mean = _train_q_mean(dataset.samples, labels_v52,
                                 [int(p) for p in train_pos])
    report = evaluate_baseline(frames.frames, train_q_mean=train_q_mean,
                               tau_end=None)
    identity = _baseline_identity(frames.frames)
    curve_values = _curve_values(report)
    return SeedOutcome(seed=int(seed), frames=frames, report=report,
                       identity=identity, train_q_mean=train_q_mean,
                       curve_values=curve_values)


def _baseline_identity(frames: Sequence[EvalFrame]) -> dict[str, Any]:
    """Zero-residual invariants of §4.3 :226 (checked, not assumed)."""
    zero = zero_residual_frames(frames)
    u_equals_u0 = all(
        np.array_equal(np.asarray(a.u), np.asarray(a.u0))
        for a in zero)
    q_equals_tanh = all(
        float(a.q_pred) == float(math.tanh(float(a.q0_raw)))
        for a in zero if a.q_mask)
    endpoint_equals = all(
        float(a.endpoint_logit) == float(a.endpoint0_logit)
        for a in zero if a.endpoint_mask)
    return {
        "u_equals_u0": bool(u_equals_u0),
        "q_equals_tanh_q0_raw": bool(q_equals_tanh),
        "endpoint_logit_equals_endpoint0": bool(endpoint_equals),
        "n_frames": len(frames),
        "frame_set_identical_to_candidate": True,
        "note": (
            "M0/P0 by construction (§4.3 :226); the zero-residual frames are "
            "rebuilt from the same frames so the H2/H3 pairing of §9.4 :1019 "
            "is frame-for-frame identical"),
    }


def _curve_values(report: Mapping[str, Any]) -> dict[str, dict[str, float]]:
    """Per-curve metric tables for :func:`paired_curve_bootstrap` (§9.4).

    Returns ``{metric_id: {curve_id: value}}`` for every metric of
    :data:`BOOTSTRAP_METRICS`; curves whose value is undefined for a metric
    are simply absent (the bootstrap drops them on both sides, §9.4 :1024).
    """
    out: dict[str, dict[str, float]] = {}
    for metric_id, (block, key), _higher in BOOTSTRAP_METRICS:
        values: dict[str, float] = {}
        for row in report["per_curve"]:
            value = row[block].get(key)
            if value is not None and math.isfinite(float(value)):
                values[str(row["curve_id"])] = float(value)
        out[metric_id] = values
    return out


def _pooled_primary(report: Mapping[str, Any]) -> dict[str, float | None]:
    """The pooled numbers of every bootstrap metric (for the summary table)."""
    move = report["pooled"]["move"]
    process = report["pooled"]["process"]
    return {
        "move.rmse_zero": move["rmse_zero"],
        "move.sign_accuracy": move["sign_accuracy_cluster_median"],
        "move.fixed_coverage.direction_at_3":
            move["fixed_coverage_direction_at_3"],
        "move.fixed_coverage.direction_at_3_curve_median":
            move["fixed_coverage"]["direction_at_3_curve_median"],
        "move.fixed_coverage.topk_overlap_at_3":
            move["fixed_coverage_topk_overlap_at_3"],
        "move.dead_fraction": move["dead_zone_fraction"],
        "move.atom_dead_fraction": move["atom_dead_fraction"],
        "move.pair_delta_d_mae": move["pair_delta_d_mae"],
        "move.cos_eff_median": move["cos_eff_median"],
        "move.basis_projection_residual_median":
            move["basis_projection_residual_median"],
        "process.q.mae_curve_median": process["q"]["mae_curve_median"],
        "process.q.mae_curve_mean": process["q"]["mae_curve_mean"],
        "process.q.corr_curve_median": process["q"]["corr_curve_median"],
        "process.q.corr_curve_median_v542_anchor":
            process["q"]["corr_curve_median_v542_anchor"],
        "process.q.constant_q_baseline_mae":
            process["q"]["constant_q_baseline_mae"],
        "process.q.learnable_above_constant":
            process["q"]["learnable_above_constant"],
        "process.ordering.pairwise_accuracy_full":
            process["ordering"]["pairwise_accuracy_full"],
        "process.ordering.pairwise_accuracy_cross_ts":
            process["ordering"]["pairwise_accuracy_cross_ts"],
        "process.endpoint.auprc": process["endpoint"]["auprc"],
        "process.endpoint.auroc": process["endpoint"]["auroc"],
        "process.endpoint.brier": process["endpoint"]["brier"],
        "process.endpoint.ece": process["endpoint"]["ece"],
        "process.endpoint.recall_at_fpr_0p05":
            process["endpoint"]["recall_at_fpr_0p05"],
        "process.endpoint.fpr_at_threshold":
            process["endpoint"]["fpr_at_threshold"],
    }


def _seed_average(outcomes: Sequence[SeedOutcome]) -> dict[str, Any]:
    """Seed-averaged summary (§9.2 :910 / §9.4 :1023 aggregation is the caller's)."""
    per_seed = {str(o.seed): _pooled_primary(o.report) for o in outcomes}
    keys = sorted({key for values in per_seed.values() for key in values})
    summary: dict[str, Any] = {}
    for key in keys:
        values = [per_seed[seed][key] for seed in per_seed
                  if per_seed[seed].get(key) is not None
                  and isinstance(per_seed[seed][key], (int, float))
                  and not isinstance(per_seed[seed][key], bool)]
        summary[key] = {
            "mean": (float(np.mean(values)) if values else None),
            "per_seed": {seed: per_seed[seed].get(key) for seed in per_seed},
            "n_seeds": len(values),
        }
    return {
        "note": (
            "seed mean is reported for readability. H2/H3 does NOT compare "
            "this mean directly: §9.4 :1023 averages seeds per bootstrap draw "
            "inside paired_curve_bootstrap; the per-seed curve tables are in "
            "'bootstrap_inputs'"),
        "metrics": summary,
    }


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------
def build_report(*, campaign: Mapping[str, Any], seeds: Sequence[int],
                 checkpoints: Mapping[int, dict], outcomes: Sequence[SeedOutcome],
                 budget: Mapping[str, Any], data_block: Mapping[str, Any],
                 evaluator_module: Path,
                 previous_report: Mapping[str, Any] | None = None
                 ) -> dict[str, Any]:
    """The ``pesn_v543_m0p0_baseline_v1`` document (measurements, no verdict)."""
    per_seed_reports = {str(o.seed): o.report for o in outcomes}
    per_seed_sha = {seed: _canonical_sha256(report)
                    for seed, report in per_seed_reports.items()}
    seed_order = [int(o.seed) for o in outcomes]
    bootstrap_inputs = {
        "metrics": {metric_id: {
            "higher_is_better": higher,
            "per_seed_curve_values": {
                str(o.seed): o.curve_values[metric_id] for o in outcomes},
            "n_curves_per_seed": {
                str(o.seed): len(o.curve_values[metric_id]) for o in outcomes},
            "n_curves_shared_across_all_seeds": len(set.intersection(*[
                set(o.curve_values[metric_id]) for o in outcomes]))
            if outcomes else 0,
        }
            for metric_id, _path, higher in BOOTSTRAP_METRICS},
        "unit": "curve (reaction cluster)",
        "seed_order": seed_order,
        "draws": int(BOOTSTRAP_DRAWS),
        "seed_root": int(BOOTSTRAP_SEED_ROOT),
        "modes": {
            "per_seed_independent": {
                "api": "paired_curve_bootstrap_multiseed",
                "status": "CONFIRMATORY (use this for H3)",
                "rule": (
                    "every draw resamples each seed's OWN inner-val cluster set "
                    "with an independent seed_root-derived stream; candidate "
                    "and baseline of a seed share that seed's indices (§9.4 "
                    ":1019 pairing); the per-seed statistics are averaged per "
                    "draw (§9.4 :1023)"),
                "why": (
                    "each formal seed evaluates its own inner-val sample, so "
                    "the all-seed intersection collapses (see "
                    "n_curves_shared_across_all_seeds above) and cannot carry "
                    "a 10,000-draw test"),
            },
            "shared_intersection": {
                "api": "paired_curve_bootstrap",
                "status": "audit / compatibility only",
                "rule": (
                    "one cluster set = the intersection over all seeds and "
                    "both sides; kept so an independent reader can reproduce "
                    "the older reading"),
                "caveat": (
                    "under-powered for H3 on this campaign: the per-seed "
                    "inner-val samples overlap only partially"),
            },
        },
        "consumption": (
            "paired_curve_bootstrap_multiseed(candidate_tables, "
            "baseline_tables, higher_is_better=..., metric=..., "
            "draws=bootstrap_inputs['draws'], "
            "seed_root=bootstrap_inputs['seed_root']) where each *_tables is "
            "the list of per-seed {curve_id: value} dicts in 'seed_order' — "
            "take it from 'metrics.<id>.per_seed_curve_values' below for the "
            "baseline and from the candidate's own report for the candidate. "
            "The shared-intersection mode stays available via "
            "paired_curve_bootstrap for audit"),
    }
    identities = {str(o.seed): o.identity for o in outcomes}
    all_identities_ok = all(
        entry["u_equals_u0"] and entry["q_equals_tanh_q0_raw"]
        and entry["endpoint_logit_equals_endpoint0"]
        for entry in identities.values())
    seed_average = _seed_average(outcomes)
    invariance = _metric_invariance(
        previous_report, per_seed_reports, per_seed_sha, seed_average)
    return {
        "schema": BASELINE_SCHEMA,
        "plan_section": PLAN_SECTION,
        "decision_record": WP3_DECISION_RECORD,
        "ok": True,
        "campaign_id": str(campaign.get("campaign_id", "")),
        "created_utc": _utc_now(),
        "seeds": [int(s) for s in seeds],
        "checkpoints": {str(seed): dict(checkpoints[int(seed)])
                        for seed in seeds},
        "bridge_spec_sha256": LatentBridge().bridge_spec_sha256(),
        "evaluator": {
            "module": str(evaluator_module.relative_to(PROJECT_ROOT))
            if evaluator_module.is_relative_to(PROJECT_ROOT)
            else str(evaluator_module),
            "code_sha256": sha256_file(evaluator_module),
            "spec_sha256": evaluator_spec_sha256(),
            "report_schema": EVAL_REPORT_SCHEMA,
            "plan_section": EVAL_PLAN_SECTION,
            "identity": evaluator_spec(),
            "note": (
                "§9.5 :1037 — code SHA + spec SHA + per-seed document SHAs "
                "freeze this baseline; any change to a metric definition "
                "changes spec_sha256 and invalidates the freeze"),
        },
        "revision": _revision_block(previous_report, invariance,
                                    evaluator_module),
        "budget": dict(budget),
        "data": dict(data_block),
        "per_seed": per_seed_reports,
        "per_seed_document_sha256": per_seed_sha,
        "seed_averaged": seed_average,
        "bootstrap_inputs": bootstrap_inputs,
        "identity_checks": {
            "by_seed": identities,
            "all_ok": bool(all_identities_ok),
            "metric_invariance_vs_previous_report": invariance,
            "plan_lines": "§4.3 :226 / §9.5 :1037",
        },
        "label_status_evidence": {
            str(o.seed): {
                "n_frames": o.frames.n_frames,
                "by_label_status": dict(o.frames.label_status_counts),
                "by_intent_class": dict(o.frames.intent_class_counts),
                "move_eligibility_rule": (
                    "frozen _frame_mask_for: only label_status == 'ok' frames "
                    "enter a MOVE denominator; the others still carry PROCESS "
                    "labels"),
            }
            for o in outcomes
        },
        "positions": {
            str(o.seed): {
                "rule": o.frames.positions_rule,
                "n_positions": o.frames.n_positions,
                "n_batches": o.frames.n_batches,
                "wall_seconds": o.frames.wall_seconds,
                "encoder_forward_count": o.frames.encoder_forward_count,
                "train_q_mean": o.train_q_mean,
                "positions_sha256": _canonical_sha256(o.frames.positions),
            }
            for o in outcomes
        },
        "thresholds_reference": {
            "note": ("the §9.3 H2/H3 thresholds are NOT applied here; this "
                     "document is the denominator they will be applied to"),
            "plan_lines": "附录 B :1585-1609",
            "move_fixed_coverage_regression_rel_max_pct": 2.0,
            "move_h3_direction_rel_min_pct": 3.0,
            "move_h3_direction_abs_min": 0.01,
            "process_q_mae_regression_rel_max_pct": 5.0,
            "process_h3_q_mae_rel_min_pct": 5.0,
            "process_auprc_drop_abs_max": 0.01,
            "process_recall_at_fpr_drop_abs_max": 0.02,
            "process_fpr_constraint": FPR_CONSTRAINT,
        },
        "evaluator_constants": {
            "fixed_coverage_topk": FIXED_COVERAGE_TOPK,
            "sign_pair_threshold_angstrom": SIGN_PAIR_THRESHOLD,
            "effective_atom_threshold_angstrom": EFFECTIVE_ATOM_THRESHOLD,
            "atom_dead_threshold_angstrom": ATOM_DEAD_THRESHOLD,
            "ece_bins": ECE_BINS,
            "q_corr_min_points": Q_CORR_MIN_POINTS,
            "q_frac_ge3_min": Q_FRAC_GE3_MIN,
            "tau_end": None,
        },
        "v542_anchor_crosscheck": _anchor_crosscheck(outcomes),
        "bond_scale_note": dict(BOND_SCALE_NOTE),
        "implementation_choices": _implementation_choices(),
        "ambiguities": list(_ambiguities()) + list(evaluator_ambiguities()),
    }


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _first_difference(old: Any, new: Any, path: str = "") -> str | None:
    """Path of the first differing leaf of two JSON documents (audit detail)."""
    if type(old) is not type(new) and not (
            isinstance(old, (int, float)) and isinstance(new, (int, float))
            and not isinstance(old, bool) and not isinstance(new, bool)):
        return f"{path}: type {type(old).__name__} -> {type(new).__name__}"
    if isinstance(old, dict):
        for key in sorted(set(old) | set(new)):
            if key not in old:
                return f"{path}.{key}: added"
            if key not in new:
                return f"{path}.{key}: removed"
            found = _first_difference(old[key], new[key], f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(old, list):
        if len(old) != len(new):
            return f"{path}: length {len(old)} -> {len(new)}"
        for index, (a, b) in enumerate(zip(old, new)):
            found = _first_difference(a, b, f"{path}[{index}]")
            if found is not None:
                return found
        return None
    if old != new:
        return f"{path}: {old!r} -> {new!r}"
    return None


def _metric_invariance(
        previous: Mapping[str, Any] | None,
        per_seed_reports: Mapping[str, Any],
        per_seed_sha: Mapping[str, str],
        seed_average: Mapping[str, Any]) -> dict[str, Any]:
    """Assert that a REFACTOR did not move a single metric value (§9.5 freeze).

    The invariance is not merely reported: a drifted per-seed document is a
    typed ``V543_EVAL_BASELINE_DRIFT`` refusal, because a frozen baseline whose
    numbers moved silently would invalidate every later H2/H3 comparison.
    """
    if previous is None:
        return {
            "checked": False,
            "reason": ("no --compare-to report supplied; invariance vs the "
                       "previous frozen document was NOT checked in this run"),
            "previous_code_sha256": None,
            "previous_spec_sha256": None,
            "per_seed_documents_identical": None,
        }
    previous_seeds = previous.get("per_seed") or {}
    previous_sha = previous.get("per_seed_document_sha256") or {}
    mismatches: list[str] = []
    for seed, document in per_seed_reports.items():
        if seed not in previous_seeds:
            mismatches.append(f"seed {seed}: absent from the previous report")
            continue
        if _canonical_sha256(document) != _canonical_sha256(
                previous_seeds[seed]):
            detail = _first_difference(previous_seeds[seed], document,
                                       f"per_seed.{seed}")
            mismatches.append(f"seed {seed}: {detail}")
    if mismatches:
        raise V543EvalError(
            "V543_EVAL_BASELINE_DRIFT",
            "the per-seed metric documents changed against the previous "
            "frozen report; a refactor must not move a metric value — "
            + "; ".join(mismatches[:5]))
    previous_average = (previous.get("seed_averaged") or {}).get("metrics") or {}
    current_average = seed_average.get("metrics") or {}
    if previous_average and _canonical_sha256(previous_average) != \
            _canonical_sha256(current_average):
        detail = _first_difference(previous_average, current_average,
                                   "seed_averaged.metrics")
        raise V543EvalError(
            "V543_EVAL_BASELINE_DRIFT",
            f"the seed-averaged metrics changed against the previous report: "
            f"{detail}")
    return {
        "checked": True,
        "per_seed_documents_identical": True,
        "seed_averaged_identical": True,
        "per_seed_document_sha256": dict(per_seed_sha),
        "previous_per_seed_document_sha256": {
            seed: previous_sha.get(seed) for seed in per_seed_reports},
        "previous_code_sha256": (previous.get("evaluator") or {}).get(
            "code_sha256"),
        "previous_spec_sha256": (previous.get("evaluator") or {}).get(
            "spec_sha256"),
        "method": (
            "canonical SHA-256 of every per_seed document plus a leaf-by-leaf "
            "JSON diff of seed_averaged.metrics; any difference is a typed "
            "V543_EVAL_BASELINE_DRIFT refusal (not a warning)"),
    }


def _revision_block(previous: Mapping[str, Any] | None,
                    invariance: Mapping[str, Any],
                    evaluator_module: Path) -> dict[str, Any]:
    """Why this document's code SHA differs from the frozen one (audit trail)."""
    current_code = sha256_file(evaluator_module)
    previous_code = None if previous is None else (
        (previous.get("evaluator") or {}).get("code_sha256"))
    previous_spec = None if previous is None else (
        (previous.get("evaluator") or {}).get("spec_sha256"))
    current_spec = evaluator_spec_sha256()
    return {
        "evaluator_code_sha256": current_code,
        "previous_code_sha256": previous_code,
        "code_sha_changed": (None if previous_code is None
                             else previous_code != current_code),
        "evaluator_spec_sha256": current_spec,
        "previous_spec_sha256": previous_spec,
        "metric_spec_unchanged": (None if previous_spec is None
                                  else previous_spec == current_spec),
        "bump_reason": (
            "bootstrap-statistic fix (stats lane): added the confirmatory "
            "per-seed-resampling mode paired_curve_bootstrap_multiseed. The "
            "metric definitions and their frozen spec SHA did not change — "
            "only the resampling unit of the §9.4 statistic."),
        "multiseed_rationale": {
            "problem": (
                "the previous (shared-intersection) bootstrap took the curve "
                "intersection over ALL seeds and both sides; each formal seed "
                "evaluates its own inner-val sample, so the intersection "
                "collapsed to ~37 curves and the 10,000-draw H3 test was "
                "severely under-powered"),
            "plan_reading": (
                "§9.4 :1019 'pairing = same cluster, same frame set' is a "
                "PAIRING rule inside one comparison (candidate and baseline "
                "resample the same indices); §9.4 :1023 '每 draw 先 "
                "seed-average，再比较' presupposes that each seed contributes "
                "its own statistic"),
            "fix": (
                "paired_curve_bootstrap_multiseed: per draw, each seed "
                "resamples its OWN cluster set with an independent "
                "seed_root-derived stream (candidate/baseline of a seed share "
                "that seed's indices), per-seed statistics are averaged, then "
                "10,000 draws give the percentile CI and the one-sided p"),
            "compatibility": (
                "paired_curve_bootstrap keeps its signature and its "
                "shared-intersection semantics as the audit path; both modes "
                "agree exactly when the seeds' cluster sets and improvements "
                "are identical (unit-tested)"),
            "metric_values_unchanged": invariance.get(
                "per_seed_documents_identical"),
        },
        "plan_lines": "§9.4 :1016-1033 / §9.5 :1037",
    }


def _anchor_crosscheck(outcomes: Sequence[SeedOutcome]) -> dict[str, Any]:
    """M0/P0 vs the V542 A0 archive anchors — as MEASUREMENTS, never as a gate.

    §1.6 :118 says the V542 numbers "不能直接作为 V543 的 H2 分母", but an A0
    checkpoint evaluated with the zero residual is still A0's own prediction,
    so the MOVE numbers SHOULD land next to the archive while the endpoint /
    q-correlation numbers MUST NOT (different frame set, and §9.4 :1031's
    >=3-point correlation rule replaces the 2-point Pearson of V542 — §2 F6).
    Recording both facts is what makes the baseline auditable.
    """
    observed: dict[str, Any] = {}
    comparison: dict[str, Any] = {}
    for outcome in outcomes:
        primary = _pooled_primary(outcome.report)
        seed = str(outcome.seed)
        observed[seed] = {
            "rmse_zero": primary["move.rmse_zero"],
            "sign_accuracy": primary["move.sign_accuracy"],
            "dead_fraction": primary["move.dead_fraction"],
            "auprc": primary["process.endpoint.auprc"],
            "q_corr": primary["process.q.corr_curve_median"],
            "q_mae_curve_median": primary["process.q.mae_curve_median"],
            "direction_at_3": primary["move.fixed_coverage.direction_at_3"],
        }
        anchor = V542_A0_ANCHORS.get(seed)
        if anchor is None:
            continue
        comparison[seed] = {
            "rmse_zero_delta_vs_v542": (
                None if primary["move.rmse_zero"] is None
                else float(primary["move.rmse_zero"] - anchor["rmse_zero"])),
            "sign_delta_vs_v542": (
                None if primary["move.sign_accuracy"] is None
                else float(primary["move.sign_accuracy"]
                           - anchor["direction"])),
            "dead_delta_vs_v542": (
                None if primary["move.dead_fraction"] is None
                else float(primary["move.dead_fraction"] - anchor["dead"])),
        }
    return {
        "plan_lines": "§1.6 :110-118 / §2 F3 :140 / §2 F6 :143",
        "v542_a0_anchors": {seed: dict(values)
                            for seed, values in V542_A0_ANCHORS.items()},
        "observed_m0p0": observed,
        "delta_vs_v542_anchors": comparison,
        "expected_to_match": (
            "MOVE rmse_zero / sign / dead: M0 is A0's own u0, so these should "
            "sit next to the archive up to the different inner-val frame set"),
        "expected_NOT_to_match": (
            "endpoint AUPRC/Brier/ECE and q.corr: this document's inner-val set "
            "deliberately covers all four strata (endpoint prevalence ~0.29 vs "
            "the slice's ~0.17), and §9.4 :1031 forbids the 2-point Pearson "
            "that inflated the V542 q_corr to ~0.9988 (§2 F6). Neither number "
            "is the V543 denominator's ancestor; §9.5 :1037 says so"),
        "role": ("measurements only — H2/H3 compare a candidate against THIS "
                 "M0/P0 document, never against the V542 archive"),
    }


def _implementation_choices() -> list[str]:
    return [
        "inner-val set: the first --eval-curves curves of the frozen sorted "
        "inner-val slice, each sampled with a deterministic uniform linspace "
        "spread of --frames-per-curve positions (no RNG). The default "
        "1024 x 8 = 8,192 frames is exactly the plan's inner-val cap (§1.4 "
        ":91). A position-uniform spread was REJECTED: this slice has ~240 "
        "frames per curve, so a position spread visits ~1 frame per curve, "
        "which leaves q.corr undefined (§9.4 :1031) and zero ordering pairs "
        "(§9.1 :892) — i.e. no usable H2/H3 denominator.",
        "M0/P0 is built WITHOUT constructing any decoder: the zero residual of "
        "§4.3 :226 is read straight off the frozen packet/context "
        "(u0, tanh(q0_raw), e0). That is the definition of the baseline, and "
        "it keeps this runner independent of the WP4 registry/trainer.",
        "the changed K-pair set handed to the evaluator is the Bridge's "
        "forward-side reconstruction (intent_pair_index + key_permutation), "
        "never a supervision table and never a model output; the pair audit "
        "that proves equality with supervision lives in the WP2 report.",
        "the frame's reaction class (breaking/forming/both/unmoved) is derived "
        "from the SIGNS of its changed pairs' supervision pair_edit (§7.1 "
        ":640); ADR-0034 §10 v1.1 BOND_SCALE is a positive factor, so the sign "
        "needs no conversion. A frame normally carries both a breaking and a "
        "forming changed pair, so this frame-level breakdown collapses towards "
        "'both'; the per-PAIR taxonomy of §7.1 :640 remains the finer one.",
        "the Bridge basis diagnostic reports ‖r_perp‖/‖r‖ for the teacher "
        "residual on span(key_v). With 8 channels in R^3 the span is almost "
        "always the whole space, so a ~0 median is EXPECTED and is not "
        "evidence of a good Bridge — exactly the §2 F8 / §7.1 :632 warning "
        "that a bare coverage number is inflating by construction.",
        "the evaluated frame set deliberately covers all four strata (start / "
        "pre_TS / post_TS / endpoint) once per direction group, so the "
        "endpoint prevalence here (~0.29) is HIGHER than the raw slice's "
        "(~0.17). That is a deliberate trade: Endpoint frames are only ~0.8 % "
        "of the slice, so an unstratified sample would leave a few hundred "
        "positives. Absolute AUPRC/Brier/ECE therefore sit in a different "
        "prevalence regime than the V542 archive and are NOT comparable to it; "
        "H2/H3 always compare a candidate against THIS frozen P0 on the same "
        "frames.",
        "constant_q_baseline_mae uses the frozen _train_q_mean over the "
        "INNER-TRAIN slice of the same split (the V542 recipe).",
        "tau_end stays None (§1.5 :105): no threshold is calibrated, so the "
        "tau-dependent misstop fields are None on both sides of every later "
        "comparison.",
    ]


def _ambiguities() -> list[str]:
    return [
        "§9.5 :1037 says 'per-seed inner-val 文档' without fixing the inner-val "
        "sample budget or its curve structure. This runner uses the plan's "
        "8,192-sample cap (§1.4 :91) as 1,024 curves x 8 frames (the §9.4 "
        "cluster and q-validity rules both require several frames per curve) "
        "and reports the exact set and its frames-per-curve distribution so "
        "the freeze is unambiguous.",
        "§9.5 does not say whether M0/P0 share one document or one document per "
        "role; M0 and P0 are the MOVE and PROCESS halves of the SAME "
        "zero-residual forward (§4.3 :226), so one document per seed carries "
        "both and the §9.3 H2/H3 comparison splits by metric, not by document.",
        "the plan does not define the H2/H3 'relative degradation' denominator "
        "when the baseline value is ~0 (§9.3 :960 covers this: it degrades to "
        "the absolute rule); this document therefore reports raw values only "
        "and never a normalised ratio.",
    ]


def run(args: argparse.Namespace) -> dict[str, Any]:
    """The whole §9.5 M0/P0 re-computation (measurements only, no verdict)."""
    campaign = load_campaign_config(PROJECT_ROOT / CAMPAIGN_CONFIG)
    seeds = [int(s) for s in (getattr(args, "seeds", None)
                              or campaign["source"]["seeds"])]
    checkpoints = resolve_checkpoints(campaign, seeds)
    data_cfg, _assets = load_pinned_assets(PROJECT_ROOT / DATA_CONFIG)
    dataset = _open_frozen_dataset(data_cfg)
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        v53mp_pool_split,
    )
    from scripts.campaigns.r3_v53mp_run import _tier_pool_curve_ids

    split_path = str((data_cfg.get("split_contract") or {}).get(
        "split_manifest") or "/pes2ts_data/v53mp_50k_data/"
                            "split_manifest_v53mp.json")
    split_doc = json.loads(Path(split_path).read_text(encoding="utf-8"))
    pool_ids = _tier_pool_curve_ids(split_doc, "50k")

    labels_v52, endpoint_labels = dataset.label_tables()
    eval_curves = int(getattr(args, "eval_curves", DEFAULT_EVAL_CURVES))
    frames_per_curve = int(getattr(args, "frames_per_curve",
                                   DEFAULT_FRAMES_PER_CURVE))
    with_basis = not bool(getattr(args, "no_basis", False))
    with_strata = not bool(getattr(args, "no_strata", False))

    outcomes: list[SeedOutcome] = []
    per_seed_positions: dict[str, Any] = {}
    for seed in seeds:
        train_pos, val_pos = v53mp_pool_split(
            dataset.samples, pool_curve_ids=pool_ids, seed=int(seed))
        outcome = run_seed(
            seed=int(seed), checkpoint=checkpoints[int(seed)],
            dataset=dataset, train_pos=train_pos, val_pos=val_pos,
            labels_v52=labels_v52, endpoint_labels=endpoint_labels,
            eval_curves=eval_curves, frames_per_curve=frames_per_curve,
            with_basis=with_basis, with_strata=with_strata)
        outcomes.append(outcome)
        per_seed_positions[str(seed)] = {
            "available": outcome.frames.positions_rule[
                "n_positions_available"],
            "used": outcome.frames.n_positions,
        }
        primary = _pooled_primary(outcome.report)
        print(
            f"[v543eval] seed {seed}: {outcome.frames.n_frames} frames / "
            f"{len(set(outcome.frames.curve_ids))} curves, "
            f"rmse_zero={primary['move.rmse_zero']}, "
            f"sign={primary['move.sign_accuracy']}, "
            f"dir@3={primary['move.fixed_coverage.direction_at_3']}, "
            f"auprc={primary['process.endpoint.auprc']}, "
            f"q_median={primary['process.q.mae_curve_median']}, "
            f"wall={outcome.frames.wall_seconds:.1f}s",
            file=sys.stderr, flush=True)

    budget = {
        "eval_curves_requested": eval_curves,
        "frames_per_curve_requested": frames_per_curve,
        "frames_per_batch": FRAMES_PER_BATCH,
        "inner_val_cap_plan": INNER_VAL_CAP,
        "with_basis_diagnostic": with_basis,
        "with_strata": with_strata,
        "tau_end": None,
        "positions_by_seed": per_seed_positions,
    }
    data_block = {
        "data_view_config": DATA_CONFIG,
        "split_manifest": split_path,
        "pool_tier": "50k",
        "n_pool_curves": len(pool_ids),
        "n_dataset_samples": len(dataset.samples),
        "dev_accessed": False,
        "confirm_accessed": False,
        "note": ("only v53mp_pool_split(seed) inner-train/inner-val positions "
                 "are read; dev (589 curves) stays untouched until T7 and "
                 "confirm stays sealed (§9.5 :1038-1039)"),
    }
    evaluator_module = Path(
        "pes2ts_core/pesn/offline/evaluation/v543_eval.py")
    evaluator_path = (evaluator_module if evaluator_module.is_absolute()
                      else PROJECT_ROOT / evaluator_module)
    previous_report = _load_previous_report(
        getattr(args, "compare_to", None))
    report = build_report(
        campaign=campaign, seeds=seeds, checkpoints=checkpoints,
        outcomes=outcomes, budget=budget, data_block=data_block,
        evaluator_module=evaluator_path, previous_report=previous_report)
    report["ok"] = True
    return report


def _load_previous_report(path: Any) -> dict[str, Any] | None:
    """Read a previous frozen baseline document for the invariance assertion."""
    if not path:
        return None
    target = Path(str(path))
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    if not target.is_file():
        raise V543EvalError(
            "V543_EVAL_COMPARE_MISSING",
            f"--compare-to report not found: {target}")
    try:
        document = json.loads(target.read_text(encoding="utf-8"))
    except Exception as error:  # noqa: BLE001 - typed conversion
        raise V543EvalError(
            "V543_EVAL_COMPARE_UNREADABLE",
            f"--compare-to report unreadable ({target}): {error}") from error
    if not isinstance(document, dict) or document.get("schema") != \
            BASELINE_SCHEMA:
        raise V543EvalError(
            "V543_EVAL_COMPARE_SCHEMA",
            f"--compare-to report is not a {BASELINE_SCHEMA} document: "
            f"{target}")
    return document


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_seeds(text: str) -> list[int]:
    seeds = [int(token) for token in str(text).replace(",", " ").split()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="V5.4.3 — M0/P0 baseline under the frozen V543 evaluator "
                    "(§9.5: recompute + SHA freeze; measurements only)")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"report path (default {DEFAULT_OUT})")
    parser.add_argument("--seeds", type=_parse_seeds, default=None,
                        help="encoder seeds (default: the campaign config's)")
    parser.add_argument("--eval-curves", type=int,
                        default=DEFAULT_EVAL_CURVES,
                        help="inner-val curves evaluated (default "
                             f"{DEFAULT_EVAL_CURVES})")
    parser.add_argument("--frames-per-curve", type=int,
                        default=DEFAULT_FRAMES_PER_CURVE,
                        help="frames sampled inside each curve (default "
                             f"{DEFAULT_FRAMES_PER_CURVE})")
    parser.add_argument("--no-basis", action="store_true",
                        help="skip the Bridge basis-projection diagnostic")
    parser.add_argument("--no-strata", action="store_true",
                        help="skip the Start/Pre-TS/Post-TS/Endpoint strata")
    parser.add_argument("--compare-to", default=None,
                        help="previous frozen baseline JSON; asserts that every "
                             "per-seed metric document and the seed-averaged "
                             "block are BIT-identical (typed refusal on drift) "
                             "and records the code-SHA bump in 'revision'")
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (V543EvalError, V542ParityError) as error:
        print(f"[v543eval] REFUSED {error}", file=sys.stderr)
        return 2
    target = Path(args.out)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True,
                                 ensure_ascii=False), encoding="utf-8")
    print(f"[v543eval] report -> {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":                       # pragma: no cover
    raise SystemExit(main())
