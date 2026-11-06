"""V5.4.3 — the frozen-identity V543 evaluator (plan §9.1/§9.2/§9.4/§9.5).

WHAT THIS MODULE IS
-------------------
The single measurement instrument of the V5.4.3 campaign: the evaluator under
which M0/P0 are re-computed and SHA-frozen (§9.5), and under which every
candidate is compared for H2/H3 (§9.3).  It is **pure**: same inputs → same
outputs, bit-for-bit (float64 numpy only), so a report can be SHA-frozen and a
re-run re-derived.  No file IO, no dataset access, no checkpoint loading, no
model.  The runner that feeds it (``scripts/agent_ops/v543_eval.py``) owns all
of that; this module only ever sees explicitly-passed arrays.

Inputs (per frame, :class:`EvalFrame`)
--------------------------------------
* **model side** — ``u`` (MOVE displacement, Å, ``u = u0 + δu`` of §4.3),
  ``q_pred`` (``q = tanh(q0_raw + δq_raw)``), ``endpoint_logit``
  (``e = e0 + δe``, never double-sigmoid'ed §4.3);
* **teacher side** — ``y`` (``y_teacher``), ``q_true``/``q_mask``,
  ``endpoint_label``/``endpoint_mask``, ``is_start_row``/``is_ts_row``;
* **grouping** — ``curve_id`` (bootstrap cluster unit, §9.4), ``direction``
  (the V542 ``(curve, intent direction)`` ordering group), ``stratum`` and
  ``intent_class`` (§9.2's two stratified breakdowns);
* **frozen pair terms** — ``pair_slots [P,2]`` + ``pair_offsets [P,3]``
  (``r_l = x_j − x_i`` of the CURRENT frame), i.e. the changed K-pair set;
* **baseline side** (optional, for :func:`evaluate_baseline`) — ``u0``,
  ``q0_raw``, ``endpoint0_logit``; and ``key_v [K,8,3]`` for the Bridge basis
  diagnostic of §9.1.

Supervision discipline (§5.6 :420–425)
--------------------------------------
Every supervision tensor above enters as an EXPLICIT, SEPARATE argument of an
offline analysis function.  Nothing here builds a ``DecoderContextV1``, calls
an encoder, or writes a supervision key into a model input.  Conversely the
evaluator never reads dev/confirm: it cannot — it only sees what it is handed.

The metrics (plan §9.1 table :872–894, formulas §9.2 :896–912)
--------------------------------------------------------------
MOVE
    ``rmse_zero`` (§9.1 :876) — per-curve ``sqrt(mean_f mse_f) /
    sqrt(mean_f mse0_f)`` with ``mse_f = mean_k ‖u−y‖²`` and
    ``mse0_f = mean_k ‖y‖²``; pooled = **median over curves** (the V542
    ``key_rmse_over_zero.median_rmse_over_zero`` leaf) + mean.
    ``sign_accuracy`` (§9.1 :877) — frozen V542 G5.3 rule on teacher-
    significant pairs (``|Δd_l(y)| ≥ 0.005 Å``; a predicted
    ``|Δd_l(u)| < 0.005 Å`` counts INCORRECT), per-curve frame mean, pooled
    **median over curves** (``sre_and_sign.sign_accuracy_cluster_median``).
    ``dead_fraction`` (§9.1 :878) — both readings are reported:
    ``dead_zone_fraction`` (the V542 leaf: teacher-active pairs whose
    predicted ``|Δd|`` stays in the dead zone, global ratio) and
    ``atom_dead_fraction`` (the plan's literal wording: share of key atoms
    with ``‖u_k‖ < 0.005 Å``).
    ``fixed_coverage.direction_at_3`` (§9.2 :904–906) — **the new independent
    primary**: for every frame with ≥1 changed K-pair, ``k = min(3,n_pairs)``,
    ``T_k`` = the teacher's ``k`` largest ``|Δd_y|`` pairs (ties by pair
    index), value ``= mean_{l∈T_k} [sign(Δd_u[l]) == sign(Δd_y[l]) AND
    |Δd_u[l]| ≥ 0.005 Å]``.  Aggregation: frame mean → curve mean → seed mean
    (:910); this module emits the curve rows and both curve-mean and
    curve-median pooled values.
    ``fixed_coverage.topk_overlap_at_3`` (§9.2 :907) — ``|P_k ∩ T_k| / k``
    with ``P_k`` the predicted ``|Δd_u|`` top-k (diagnostic).
    ``pair_delta_d.mae`` (§9.1 :881) — ``mean |Δd_u − Δd_y|`` over the changed
    K-pairs (chemistry-action diagnostic).
    ``basis_projection_residual`` (§9.1 :882, Bridge diagnostic) — median over
    key rows of ``‖r_perp‖/‖r‖`` for the teacher residual ``r = y − u0``
    projected on ``span(key_v)`` (minimum-norm least squares, the S0-C
    convention); only when ``key_v``/``u0`` are supplied.
    ``cos_eff_median`` / ``n_effective_atoms`` — the frozen G5.2 effective-
    atom soft cosine (``‖y_k‖ > 0.02 Å``), kept per §9.1's "既有指标按方案保留".
PROCESS
    ``q.mae_curve_median`` (primary) / ``mae_curve_mean`` / ``n_eval_curves`` /
    ``median_points_per_curve`` / ``frac_curves_ge3`` / ``frac_curves_ge5``;
    ``q.corr_curve_median`` **diagnostic only** under the §9.4 q-validity rule
    (:1029–1033): a curve needs **≥3** valid q points to enter the
    correlation, and ``frac_curves_ge3 < 0.70`` raises
    ``corr_diagnostic_only = True``.  ``constant_q_baseline_mae`` /
    ``learnable_above_constant`` = the V542 §q_head leaves (pooled
    ``mean |q0_const − q_true|`` against the inner-train q mean).
    ``ordering.pairwise_accuracy_full`` / ``_cross_ts`` — the frozen V542
    ``ordering_pair_accuracy`` on ``(curve, direction)`` groups.
    ``endpoint.auprc`` / ``auroc`` / ``brier`` / ``ece`` (10 bins) /
    ``recall_at_fpr_0p05`` (§9.1 :884, 附录 B ``process_fpr_constraint =
    0.05``): the threshold maximizing recall subject to ``FPR ≤ 0.05`` on the
    SAME inner-val frame set, with the achieved FPR reported.
    ``tau_end = None`` (§1.5 :105) — no threshold metric depends on tau; the
    tau-dependent misstop fields are ``None`` exactly as in V542.

NOT re-implemented (frozen V542 口径, imported lazily)
-----------------------------------------------------
``average_precision``, ``auc_mann_whitney``, ``_sigmoid`` and
``ordering_pair_accuracy`` come from
``pes2ts_core/pesn/offline/training/key_atom_v53mp_train_v3.py`` itself (never
a copy, so the numbers cannot drift from the frozen evaluator);
``hierarchical_means`` comes from
``pes2ts_core/pesn/offline/evaluation/key_atom_protocols.py`` — the normative
V542 aggregation order (frame → branch → reaction → cluster).  The
``fixed_coverage_direction@3`` metric is **not** imported and has no V542
counterpart: V542's "direction" leaf IS ``sign_accuracy_cluster_median``
(``scripts/agent_ops/v542_stats.py`` :177–190), which is why §1.5 :108 and §2
F3 require a physically different implementation here.

ADR-0034 §10 v1.1 (BOND_SCALE)
------------------------------
``pair_edit = BOND_SCALE × forward ΔB`` with ``BOND_SCALE = 3.0``.  No metric
in this module compares a distance change ``Δd`` to ``pair_edit``; the
``/3.0`` conversion is therefore NOT applied anywhere, and
:data:`BOND_SCALE_NOTE` records that fact so a later diagnostic that *does*
compare them cannot silently forget it.

Everything here is CPU-fast and tested on synthetic tensors in
``tests/pesn/test_v543_eval.py``; the real-data certificate is
``tests/pesn/test_v543_eval_realdata.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Final, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.features.key_atom_spec_v3 import BOND_SCALE
from pes2ts_core.pesn.models.v543.contracts import (
    KEY_V_CHANNELS_V543,
    V543Error,
)
from pes2ts_core.pesn.offline.evaluation.key_atom_protocols import (
    COS_SOFT_NORM,
    EFFECTIVE_ATOM_TAU,
    G53_SIGN_PAIR_THRESHOLD,
    G53_SRE_MIN_TOTAL_CHANGE,
    hierarchical_means,
)

# ---------------------------------------------------------------------------
# typed refusal (code/detail; the WP1–WP4 taxonomy)
# ---------------------------------------------------------------------------
class V543EvalError(V543Error):
    """The frozen evaluator refused to produce a number."""


# ---------------------------------------------------------------------------
# frozen constants — every number is quoted from the plan or from the frozen
# V542 G5.2/G5.3 protocol; none is invented here
# ---------------------------------------------------------------------------
#: Plan §9.2 :903 — ``k = min(3, n_pairs)``; §9.1 :879 calls it ``@3``.
FIXED_COVERAGE_TOPK: Final[int] = 3
#: Frozen ``G53_SIGN_PAIR_THRESHOLD`` (Å) — the pair dead-zone / significance
#: threshold quoted by §9.2 :906 as ``|Δd_u| ≥ 0.005 Å``.
SIGN_PAIR_THRESHOLD: Final[float] = float(G53_SIGN_PAIR_THRESHOLD)
#: Frozen ``EFFECTIVE_ATOM_TAU`` (Å) — G5.2 effective-atom gate.
EFFECTIVE_ATOM_THRESHOLD: Final[float] = float(EFFECTIVE_ATOM_TAU)
#: Frozen ``COS_SOFT_NORM`` — the soft-norm epsilon of the G5.2 cosine.
COSINE_SOFT_NORM: Final[float] = float(COS_SOFT_NORM)
#: Frozen ``G53_SRE_MIN_TOTAL_CHANGE`` (Å) — kept for the pair-count evidence.
SRE_MIN_TOTAL_CHANGE: Final[float] = float(G53_SRE_MIN_TOTAL_CHANGE)
#: Plan 附录 B :1603 — ``process_fpr_constraint = 0.05``.
FPR_CONSTRAINT: Final[float] = 0.05
#: Plan §9.1 :886 — "``endpoint.ece`` | 校准 | 10 bins".
ECE_BINS: Final[int] = 10
#: Plan §9.4 :1031–1032 — a curve needs ≥3 valid q points before a Pearson
#: correlation may be computed at all ("每曲线 ≥3 点才可算 corr").
Q_CORR_MIN_POINTS: Final[int] = 3
#: Plan §9.4 :1032 — ``frac_curves_ge3 < 0.70`` ⇒ ``q_corr`` is diagnostic only.
Q_FRAC_GE3_MIN: Final[float] = 0.70
#: §9.4 :1031 — the ≥5-point share is reported alongside.
Q_GE5_MIN_POINTS: Final[int] = 5
#: §9.1 :878 ``dead_fraction`` literal reading ("预测位移小于阈值的比例"); the
#: plan fixes no number, so the campaign's own dead-zone constant is used and
#: the choice is reported (see :func:`ambiguities`).
ATOM_DEAD_THRESHOLD: Final[float] = SIGN_PAIR_THRESHOLD
#: V542 G5.2: a denominator below this (Å) is UNDEFINED, never coerced to 0.
DENOM_EPS: Final[float] = 1.0e-6
#: The S0-C floor under which a teacher residual has no direction (Å).
RESIDUAL_NORM_FLOOR: Final[float] = 1.0e-9
#: Plan 附录 B :1608 — ``bootstrap_draws = 10000``.
BOOTSTRAP_DRAWS: Final[int] = 10_000
#: Plan 附录 B :1609 — ``bootstrap_seed_root = 20261102``.
BOOTSTRAP_SEED_ROOT: Final[int] = 20261102
#: Plan §9.4 :1021 — percentile 95 % (2.5 / 97.5).
CI_LOW_PERCENTILE: Final[float] = 2.5
CI_HIGH_PERCENTILE: Final[float] = 97.5

#: Report schema id (naming mirrors ``pesn_v543_s0_probe_report_v1``).
REPORT_SCHEMA: Final[str] = "pesn_v543_eval_report_v1"
#: The plan lines this evaluator implements (quoted in every report).
PLAN_SECTION: Final[str] = (
    "§9.1 :872-894 / §9.2 :896-912 / §9.3 :914-1012 / §9.4 :1013-1033 / "
    "§9.5 :1035-1041 / §1.5 :105-108 / 附录 B :1585-1609")

#: Plan §9.2 :910 — the frozen aggregation chain of the new metric.
FIXED_COVERAGE_AGGREGATION: Final[str] = "frame mean -> curve mean -> seed mean"
#: Plan §9.4 :1018 — the bootstrap cluster unit.
BOOTSTRAP_UNIT: Final[str] = "curve (reaction cluster)"
#: Plan §9.4 :1019 — pairing rule.
BOOTSTRAP_PAIRING: Final[str] = (
    "same cluster, same frame set; candidate and baseline share the resample "
    "indices")
#: Plan §9.4 :1023 — seed aggregation order.
BOOTSTRAP_SEED_AGGREGATION: Final[str] = (
    "per draw seed-average FIRST, then compare (每 draw 先 seed-average，再比较)")

#: §9.2 :910 — the two stratified breakdowns the new metric is reported under.
STRATA_ORDER: Final[tuple[str, ...]] = ("start", "pre_TS", "post_TS",
                                        "endpoint")
#: §9.2 :910 / §7.1 :640 — the reaction-type taxonomy (teacher-side signs of
#: the changed pairs' ``pair_edit``).
INTENT_CLASSES: Final[tuple[str, ...]] = ("breaking", "forming", "both",
                                          "unmoved")

#: ADR-0034 §10 v1.1 — the frozen scale relation, recorded as DATA (nothing in
#: this module compares Δd to ``pair_edit``, so no conversion is applied).
BOND_SCALE_NOTE: Final[Mapping[str, Any]] = {
    "bond_scale": float(BOND_SCALE),
    "relation": "pair_edit = BOND_SCALE * forward ΔB",
    "applied_here": False,
    "why": (
        "every metric of this evaluator is a function of Δd (Å) and of the "
        "teacher displacement y (Å); pair_edit is never read, so the /3.0 "
        "conversion must NOT be applied. A future diagnostic that compares "
        "Δd against pair_edit has to divide by BOND_SCALE."),
}


# ---------------------------------------------------------------------------
# the per-frame input record
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EvalFrame:
    """One evaluated frame: model side + teacher side + grouping + pairs.

    All arrays are float64/int64 numpy on CPU.  ``pair_slots`` are indices into
    the ``u``/``y`` rows of THIS frame (0-based key slots); ``pair_offsets``
    are the CURRENT-frame difference vectors ``r_l = x_j − x_i`` (Å) of those
    pairs.  ``endpoint_label`` uses the frozen encoding: ``0``/``1`` are valid
    labels, anything else (``-1`` = masked) never enters a denominator.

    ``u0``/``q0_raw``/``endpoint0_logit`` are the zero-residual baseline
    quantities (§4.3 :226) and are only required by
    :func:`evaluate_baseline`; ``key_v`` is only required by the Bridge basis
    diagnostic.

    ``move_eligible`` carries the frozen MOVE mask rule (§4.1
    ``_frame_mask_for``: only ``label_status == "ok"`` frames have a usable
    ``y_teacher``); a terminal/masked/conflict frame stays in the frame set for
    PROCESS but may never enter a MOVE denominator.
    """

    curve_id: str
    u: np.ndarray
    y: np.ndarray
    pair_slots: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 2), dtype=np.int64))
    pair_offsets: np.ndarray = field(
        default_factory=lambda: np.zeros((0, 3), dtype=np.float64))
    direction: str = ""
    stratum: str = ""
    intent_class: str = ""
    q_pred: float | None = None
    q_true: float | None = None
    q_mask: bool = False
    endpoint_logit: float | None = None
    endpoint_label: int = -1
    endpoint_mask: bool = False
    is_start_row: bool = False
    is_ts_row: bool = False
    #: MOVE eligibility: the frozen ``_frame_mask_for`` rule — only
    #: ``label_status == "ok"`` frames carry a usable ``y_teacher`` (§4.1).
    #: Terminal/masked/conflict frames stay in the frame set because their
    #: PROCESS labels are valid, but they never enter a MOVE denominator.
    move_eligible: bool = True
    u0: np.ndarray | None = None
    q0_raw: float | None = None
    endpoint0_logit: float | None = None
    key_v: np.ndarray | None = None

    # -- validation ---------------------------------------------------------
    def check(self) -> "EvalFrame":
        """Validate shapes/dtypes/finiteness; typed refusal on any drift."""
        if not isinstance(self.curve_id, str) or not self.curve_id:
            raise V543EvalError(
                "V543_EVAL_FRAME_CURVE",
                f"curve_id must be a non-empty string, got {self.curve_id!r}")
        u = _as_array(self.u, "u", ndim=2, width=3)
        y = _as_array(self.y, "y", ndim=2, width=3)
        if u.shape != y.shape:
            raise V543EvalError(
                "V543_EVAL_FRAME_SHAPE",
                f"curve {self.curve_id!r}: u {list(u.shape)} and y "
                f"{list(y.shape)} are not row aligned")
        slots = _as_array(self.pair_slots, "pair_slots", ndim=2, width=2,
                          dtype=np.int64)
        offsets = _as_array(self.pair_offsets, "pair_offsets", ndim=2, width=3)
        if int(slots.shape[0]) != int(offsets.shape[0]):
            raise V543EvalError(
                "V543_EVAL_FRAME_SHAPE",
                f"curve {self.curve_id!r}: {int(slots.shape[0])} pair slots "
                f"vs {int(offsets.shape[0])} pair offsets")
        if int(slots.size):
            if int(slots.min()) < 0 or int(slots.max()) >= int(u.shape[0]):
                raise V543EvalError(
                    "V543_EVAL_PAIR_SLOT",
                    f"curve {self.curve_id!r}: pair slots must lie in "
                    f"[0,{int(u.shape[0])}), got "
                    f"[{int(slots.min())},{int(slots.max())}]")
            if bool((slots[:, 0] == slots[:, 1]).any()):
                raise V543EvalError(
                    "V543_EVAL_PAIR_SLOT",
                    f"curve {self.curve_id!r}: a pair has identical slots")
        if self.q_mask and (self.q_pred is None or self.q_true is None):
            raise V543EvalError(
                "V543_EVAL_Q",
                f"curve {self.curve_id!r}: q_mask=True needs q_pred and "
                "q_true")
        if self.q_mask and not (math.isfinite(float(self.q_pred))
                                and math.isfinite(float(self.q_true))):
            raise V543EvalError(
                "V543_EVAL_Q",
                f"curve {self.curve_id!r}: masked-in q values must be finite")
        if self.endpoint_mask:
            if self.endpoint_logit is None or not math.isfinite(
                    float(self.endpoint_logit)):
                raise V543EvalError(
                    "V543_EVAL_ENDPOINT",
                    f"curve {self.curve_id!r}: endpoint_mask=True needs a "
                    "finite endpoint_logit")
            if int(self.endpoint_label) not in (0, 1):
                raise V543EvalError(
                    "V543_EVAL_ENDPOINT",
                    f"curve {self.curve_id!r}: a masked-in endpoint row must "
                    f"carry label 0/1, got {self.endpoint_label!r} (§5.3)")
        if self.u0 is not None:
            base = _as_array(self.u0, "u0", ndim=2, width=3)
            if base.shape != u.shape:
                raise V543EvalError(
                    "V543_EVAL_FRAME_SHAPE",
                    f"curve {self.curve_id!r}: u0 {list(base.shape)} and u "
                    f"{list(u.shape)} are not row aligned")
        if self.key_v is not None:
            basis = np.asarray(self.key_v, dtype=np.float64)
            if basis.ndim != 3 or int(basis.shape[1]) != KEY_V_CHANNELS_V543 \
                    or int(basis.shape[2]) != 3:
                raise V543EvalError(
                    "V543_EVAL_FRAME_SHAPE",
                    f"curve {self.curve_id!r}: key_v must be "
                    f"[K,{KEY_V_CHANNELS_V543},3], got {list(basis.shape)}")
            if int(basis.shape[0]) != int(u.shape[0]):
                raise V543EvalError(
                    "V543_EVAL_FRAME_SHAPE",
                    f"curve {self.curve_id!r}: key_v has {int(basis.shape[0])} "
                    f"key rows for {int(u.shape[0])} displacement rows")
            if not bool(np.isfinite(basis).all()):
                raise V543EvalError(
                    "V543_EVAL_FRAME_NONFINITE", "key_v is non-finite")
        return self

    @property
    def n_keys(self) -> int:
        return int(np.asarray(self.u).shape[0])

    @property
    def n_pairs(self) -> int:
        return int(np.asarray(self.pair_slots).shape[0])


def _as_array(value: Any, name: str, *, ndim: int, width: int,
              dtype: Any = np.float64) -> np.ndarray:
    """Cast to a finite float64/int64 array of the stated trailing width."""
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise V543EvalError(
            "V543_EVAL_FRAME_DTYPE", f"{name} is not castable: {exc}") from exc
    if array.ndim != ndim or int(array.shape[-1]) != int(width):
        raise V543EvalError(
            "V543_EVAL_FRAME_SHAPE",
            f"{name} must be {ndim}-D with trailing width {width}, got "
            f"{list(array.shape)}")
    if dtype is np.float64 and not bool(np.isfinite(array).all()):
        raise V543EvalError(
            "V543_EVAL_FRAME_NONFINITE", f"{name} contains non-finite values")
    return array


# ---------------------------------------------------------------------------
# frozen geometry: the plan §9.2 Δd of a key pair
# ---------------------------------------------------------------------------
def exact_delta_d(u: np.ndarray, pair_slots: np.ndarray,
                  pair_offsets: np.ndarray) -> np.ndarray:
    """``Δd_l(u) = ‖r_l + u_j − u_i‖ − ‖r_l‖`` (plan §9.2 :901–902).

    Identical to the frozen ``key_atom_protocols.exact_delta_d`` (which takes
    ``pos`` + global key rows instead of ``r_l``): the plan's
    ``‖x_i + u_i − (x_j + u_j)‖ − ‖x_i − x_j‖`` equals ``‖r + u_j − u_i‖ −
    ‖r‖`` with ``r = x_j − x_i`` because a norm is invariant under negation.
    """
    u = np.asarray(u, dtype=np.float64)
    slots = np.asarray(pair_slots, dtype=np.int64)
    offsets = np.asarray(pair_offsets, dtype=np.float64)
    if int(slots.shape[0]) == 0:
        return np.zeros(0, dtype=np.float64)
    displacement = u[slots[:, 1]] - u[slots[:, 0]]
    return (np.linalg.norm(offsets + displacement, axis=-1)
            - np.linalg.norm(offsets, axis=-1))


def topk_pair_indices(values: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` largest ``|values|``, ties broken by pair index.

    Plan §9.2 :904–905 — "``T_k = 教师 |dd_y| 最大的 k 个 pair（ties 按 pair
    索引）``"; ``P_k`` uses the same rule on ``|dd_u|``.
    """
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if int(k) <= 0 or array.size == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.lexsort((np.arange(array.size, dtype=np.int64),
                        -np.abs(array)))
    return order[:min(int(k), int(array.size))].astype(np.int64)


def fixed_coverage_at_k(dd_y: np.ndarray, dd_u: np.ndarray, *,
                        k: int = FIXED_COVERAGE_TOPK,
                        threshold: float = SIGN_PAIR_THRESHOLD
                        ) -> dict[str, Any]:
    """The plan §9.2 metric pair for ONE frame.

    Returns ``direction`` (``None`` when the frame has no pair),
    ``topk_overlap``, the chosen ``T_k``/``P_k`` indices, ``k`` and the
    per-pair hit flags — the indices make the number auditable by hand.
    """
    dd_y = np.asarray(dd_y, dtype=np.float64).reshape(-1)
    dd_u = np.asarray(dd_u, dtype=np.float64).reshape(-1)
    if dd_y.size != dd_u.size:
        raise V543EvalError(
            "V543_EVAL_PAIR_SHAPE",
            f"teacher/prediction Δd sizes differ: {dd_y.size} vs {dd_u.size}")
    n_pairs = int(dd_y.size)
    if n_pairs == 0:
        return {"direction": None, "topk_overlap": None, "k": 0,
                "teacher_topk": (), "prediction_topk": (), "hits": ()}
    kk = min(int(k), n_pairs)
    teacher_top = topk_pair_indices(dd_y, kk)
    prediction_top = topk_pair_indices(dd_u, kk)
    hits = (np.sign(dd_u[teacher_top]) == np.sign(dd_y[teacher_top])) & (
        np.abs(dd_u[teacher_top]) >= float(threshold))
    overlap = len(set(teacher_top.tolist()) & set(prediction_top.tolist()))
    return {
        "direction": float(hits.mean()),
        "topk_overlap": float(overlap) / float(kk),
        "k": int(kk),
        "teacher_topk": tuple(int(i) for i in teacher_top),
        "prediction_topk": tuple(int(i) for i in prediction_top),
        "hits": tuple(bool(v) for v in hits),
    }


def soft_cosine(u: np.ndarray, y: np.ndarray, *,
                soft: float = COSINE_SOFT_NORM) -> np.ndarray:
    """Per-atom soft-norm cosine — the frozen G5.2 formula (:`key_atom_
    protocols._soft_cosine_matrix`).

    Repeated here (vectorised, same arithmetic) because the evaluator must stay
    importable without pulling the model layers; the frozen module's function
    is private and the formula is two lines, so the risk of silent drift is
    handled by ``tests/pesn/test_v543_eval.py`` asserting agreement with it.
    """
    u = np.asarray(u, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    numerator = np.sum(u * y, axis=-1)
    denominator = np.sqrt(np.sum(u * u, axis=-1) + soft * soft) * np.sqrt(
        np.sum(y * y, axis=-1) + soft * soft)
    return numerator / np.maximum(denominator, 1e-12)


def _ratio(numerator: float, denominator: float) -> float | None:
    """``num/den``; a denominator below :data:`DENOM_EPS` is UNDEFINED (§6)."""
    if not math.isfinite(denominator) or abs(denominator) < DENOM_EPS:
        return None
    return float(numerator / denominator)


def _median(values: Sequence[float | None]) -> float | None:
    vals = [float(v) for v in values
            if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _mean(values: Sequence[float | None]) -> float | None:
    vals = [float(v) for v in values
            if v is not None and math.isfinite(float(v))]
    if not vals:
        return None
    return float(np.mean(np.asarray(vals, dtype=np.float64)))


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson r; ``None`` for constant vectors / <2 points (§9.4 :1033)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or x.size != y.size:
        return None
    dx = x - x.mean()
    dy = y - y.mean()
    denominator = math.sqrt(float((dx * dx).sum()) * float((dy * dy).sum()))
    if denominator <= 0.0:
        return None
    return float(float((dx * dy).sum()) / denominator)


def _sigmoid_scalar(logit: float) -> float:
    """Sigmoid with the V542 clip (``average_precision`` side is unchanged)."""
    return float(1.0 / (1.0 + math.exp(-max(min(float(logit), 60.0),
                                            -60.0))))


# ---------------------------------------------------------------------------
# the frozen V542 math helpers (imported, never copied)
# ---------------------------------------------------------------------------
def _frozen_ordering():
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        ordering_pair_accuracy,
    )

    return ordering_pair_accuracy


def _frozen_endpoint_math():
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        average_precision,
        _sigmoid,
    )
    from pes2ts_core.pesn.offline.training.key_atom_v52_train_v3 import (
        auc_mann_whitney,
    )

    return average_precision, auc_mann_whitney, _sigmoid


# ---------------------------------------------------------------------------
# per-frame terms
# ---------------------------------------------------------------------------
@dataclass
class FrameTerms:
    """Every per-frame quantity the aggregations need (never re-derived)."""

    curve_id: str
    n_key_rows: int
    n_pairs: int
    mse: float                       # mean_k ‖u−y‖²
    mse_zero: float                  # mean_k ‖y‖²
    sign_accuracy: float | None
    n_active_pairs: int
    n_dead_pairs: int
    fixed_direction: float | None
    topk_overlap: float | None
    dd_abs_error_sum: float
    dd_count: int
    cos_values: np.ndarray
    n_dead_atoms: int
    basis_residuals: np.ndarray
    stratum: str
    intent_class: str
    direction: str
    move_eligible: bool = True


def frame_terms(frame: EvalFrame, *,
                topk: int = FIXED_COVERAGE_TOPK) -> FrameTerms:
    """All per-frame quantities of one :class:`EvalFrame` (plan §9.1/§9.2).

    A frame whose ``move_eligible`` is false (``label_status != "ok"``) still
    yields its PROCESS terms but NO MOVE term: its ``y_teacher`` is not a
    teacher (§4.1), so it may never enter a MOVE denominator.
    """
    frame.check()
    u = np.asarray(frame.u, dtype=np.float64)
    y = np.asarray(frame.y, dtype=np.float64)
    slots = np.asarray(frame.pair_slots, dtype=np.int64)
    offsets = np.asarray(frame.pair_offsets, dtype=np.float64)

    if not frame.move_eligible:
        return FrameTerms(
            curve_id=str(frame.curve_id),
            n_key_rows=int(u.shape[0]),
            n_pairs=int(slots.shape[0]),
            mse=float("nan"),
            mse_zero=float("nan"),
            sign_accuracy=None,
            n_active_pairs=0,
            n_dead_pairs=0,
            fixed_direction=None,
            topk_overlap=None,
            dd_abs_error_sum=0.0,
            dd_count=0,
            cos_values=np.zeros(0),
            n_dead_atoms=0,
            basis_residuals=np.zeros(0),
            stratum=str(frame.stratum),
            intent_class=str(frame.intent_class),
            direction=str(frame.direction),
            move_eligible=False,
        )

    residual = u - y
    mse = float(np.mean(np.sum(residual * residual, axis=-1)))
    mse_zero = float(np.mean(np.sum(y * y, axis=-1)))
    y_norms = np.linalg.norm(y, axis=-1)

    # --- G5.3 sign accuracy + dead zone (frozen V542 rule) ------------------
    dd_y = exact_delta_d(y, slots, offsets)
    dd_u = exact_delta_d(u, slots, offsets)
    sign_accuracy = None
    n_active = 0
    n_dead = 0
    if dd_y.size:
        significant = np.abs(dd_y) >= SIGN_PAIR_THRESHOLD
        n_active = int(significant.sum())
        if n_active:
            correct = (np.sign(dd_u[significant])
                       == np.sign(dd_y[significant])) & (
                np.abs(dd_u[significant]) >= SIGN_PAIR_THRESHOLD)
            n_dead = int((np.abs(dd_u[significant]) < SIGN_PAIR_THRESHOLD)
                         .sum())
            sign_accuracy = float(correct.mean())

    # --- §9.2 fixed coverage @k (the NEW independent metric) ----------------
    coverage = fixed_coverage_at_k(dd_y, dd_u, k=int(topk))

    # --- effective-atom cosine (G5.2, frozen) ------------------------------
    effective = y_norms > EFFECTIVE_ATOM_THRESHOLD
    cos_values = (soft_cosine(u[effective], y[effective])
                  if bool(effective.any()) else np.zeros(0))

    # --- the plan's literal dead_fraction (‖u_k‖ < threshold) --------------
    n_dead_atoms = int((np.linalg.norm(u, axis=-1) < ATOM_DEAD_THRESHOLD).sum())

    # --- Bridge basis projection residual (§9.1 :882) ----------------------
    basis = np.zeros(0)
    if frame.u0 is not None and frame.key_v is not None:
        basis = _basis_projection_residual(
            np.asarray(frame.key_v, dtype=np.float64),
            y - np.asarray(frame.u0, dtype=np.float64))

    return FrameTerms(
        curve_id=str(frame.curve_id),
        n_key_rows=int(u.shape[0]),
        n_pairs=int(dd_y.size),
        mse=mse,
        mse_zero=mse_zero,
        sign_accuracy=sign_accuracy,
        n_active_pairs=n_active,
        n_dead_pairs=n_dead,
        fixed_direction=coverage["direction"],
        topk_overlap=coverage["topk_overlap"],
        dd_abs_error_sum=(float(np.abs(dd_u - dd_y).sum())
                          if dd_y.size else 0.0),
        dd_count=int(dd_y.size),
        cos_values=np.asarray(cos_values, dtype=np.float64),
        n_dead_atoms=n_dead_atoms,
        basis_residuals=np.asarray(basis, dtype=np.float64),
        stratum=str(frame.stratum),
        intent_class=str(frame.intent_class),
        direction=str(frame.direction),
        move_eligible=True,
    )


def _basis_projection_residual(key_v: np.ndarray,
                               residual: np.ndarray) -> np.ndarray:
    """``‖r_perp‖/‖r‖`` per key row (S0-C convention, minimum-norm LS).

    ``key_v`` is ``[K,8,3]`` (channels are basis vectors); the projection uses
    the pseudo-inverse so a rank-deficient basis is handled honestly.  Rows
    whose residual norm is at/below :data:`RESIDUAL_NORM_FLOOR` are excluded
    (their ratio is undefined — never a silent 0).
    """
    basis = key_v.transpose(0, 2, 1)          # [K,3,8] — columns are channels
    pinv = np.linalg.pinv(basis)              # [K,8,3]
    coefficients = np.einsum("kcd,kd->kc", pinv, residual)
    projection = np.einsum("kdc,kc->kd", basis, coefficients)
    perpendicular = residual - projection
    norms = np.linalg.norm(residual, axis=-1)
    keep = norms > RESIDUAL_NORM_FLOOR
    if not bool(keep.any()):
        return np.zeros(0)
    return (np.linalg.norm(perpendicular[keep], axis=-1)
            / norms[keep])


# ---------------------------------------------------------------------------
# per-curve aggregation
# ---------------------------------------------------------------------------
def _curve_mean(values: Sequence[float], curve_ids: Sequence[str],
                curve: str) -> float | None:
    """Mean of the defined frame values of one curve (V542 aggregation).

    Uses the frozen ``hierarchical_means`` with ``reaction = cluster`` and
    ``branch = 0`` — the exact identity the frozen trainer path uses
    (``evaluate_checkpoint`` passes ``reaction_id = cluster``, ``branch = 0``),
    so a per-curve value here equals the V542 per-cluster value.
    """
    picked = [(v, c) for v, c in zip(values, curve_ids)
              if c == curve and v is not None and math.isfinite(float(v))]
    if not picked:
        return None
    frame_values = np.asarray([v for v, _ in picked], dtype=np.float64)
    clusters = [c for _, c in picked]
    means = hierarchical_means(frame_values, clusters, clusters,
                               [0] * len(clusters))
    return float(means[curve])


def _endpoint_block(logits: Sequence[float], labels: Sequence[int], *,
                    compute_auprc: bool = True) -> dict[str, Any]:
    """Endpoint metrics of ONE row set (Brier/ECE/AUPRC/AUROC)."""
    scores = np.asarray([_sigmoid_scalar(v) for v in logits], dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    out: dict[str, Any] = {
        "n_rows": int(y.size),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "brier": (float(np.mean((scores - y) ** 2)) if y.size else None),
        "ece": _ece(scores, y),
        "auprc": None,
        "auroc": None,
    }
    if compute_auprc and y.size and n_pos and n_neg:
        average_precision, auc_mann_whitney, _sigmoid = _frozen_endpoint_math()
        out["auprc"] = average_precision(scores.tolist(), y.tolist())
        out["auroc"] = auc_mann_whitney(scores.tolist(), y.tolist())
    return out


def _ece(scores: np.ndarray, labels: np.ndarray, *,
         bins: int = ECE_BINS) -> float | None:
    """Expected calibration error over ``bins`` equal-width [0,1] bins.

    Plan §9.1 :886 "``endpoint.ece`` | 校准 | 10 bins".  Empty bins are
    skipped (they carry no probability mass); the weighted absolute gap
    ``Σ (n_b/n) |acc_b − conf_b|`` is the ECE.
    """
    if scores.size == 0:
        return None
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    index = np.clip(np.digitize(scores, edges[1:-1], right=False), 0,
                    int(bins) - 1)
    total = float(scores.size)
    ece = 0.0
    for b in range(int(bins)):
        mask = index == b
        count = int(mask.sum())
        if count == 0:
            continue
        accuracy = float(labels[mask].mean())
        confidence = float(np.mean(scores[mask]))
        ece += (float(count) / total) * abs(accuracy - confidence)
    return float(ece)


def recall_at_fpr(scores: Sequence[float], labels: Sequence[int], *,
                  fpr_constraint: float = FPR_CONSTRAINT) -> dict[str, Any]:
    """Operating point of plan §9.1 :884 — "选阈值使 FPR≤0.05".

    The threshold is chosen on the SAME row set it is reported on (inner-val,
    §9.1 :884 / §9.5 :1040: tau is calibrated once on inner-val, never on dev).
    Among every distinct score threshold whose achieved FPR is ``<=
    fpr_constraint`` the one with the highest recall wins; ties go to the
    HIGHER threshold (the more conservative operating point), which makes the
    choice deterministic.
    """
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    out: dict[str, Any] = {
        "fpr_constraint": float(fpr_constraint),
        "threshold": None,
        "recall": None,
        "fpr": None,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "rule": ("maximise recall subject to achieved FPR <= constraint on "
                 "the same frame set; ties -> higher threshold"),
    }
    if n_pos == 0 or n_neg == 0 or s.size == 0:
        return out
    candidates = np.unique(s)[::-1]
    best: tuple[float, float, float] | None = None      # (recall, thr, fpr)
    for threshold in candidates:
        predicted = s >= float(threshold)
        tp = int((predicted & (y == 1)).sum())
        fp = int((predicted & (y == 0)).sum())
        fpr = float(fp) / float(n_neg)
        if fpr > float(fpr_constraint) + 1e-12:
            continue
        recall = float(tp) / float(n_pos)
        if best is None or recall > best[0] + 1e-12 or (
                abs(recall - best[0]) <= 1e-12
                and float(threshold) > best[1]):
            best = (recall, float(threshold), fpr)
    if best is None:
        # the empty set always satisfies FPR <= constraint
        best = (0.0, float(candidates.max()) + 1.0, 0.0)
    out["threshold"], out["recall"], out["fpr"] = best[1], best[0], best[2]
    return out


# ---------------------------------------------------------------------------
# the evaluator
# ---------------------------------------------------------------------------
def evaluate_v543(
        frames: Sequence[EvalFrame],
        *,
        train_q_mean: float | None = None,
        tau_end: float | None = None,
        topk: int = FIXED_COVERAGE_TOPK,
        fpr_constraint: float = FPR_CONSTRAINT,
        corr_min_points: int = Q_CORR_MIN_POINTS,
) -> dict[str, Any]:
    """Evaluate one role (M0/P0 baseline or a candidate) on one frame set.

    Args:
        frames: the evaluated frames, all from one seed's inner-val slice.
        train_q_mean: the inner-train mean of ``q`` (the V542
            ``constant_q_baseline_mae`` reference, ``_train_q_mean``); it is
            the ONLY quantity this evaluator needs from the train split and it
            is passed explicitly.
        tau_end: the endpoint threshold.  ``None`` is the §1.5 :105 frozen
            semantics ("``tau_end=None``") — every tau-dependent field is
            ``None`` and no threshold is invented.
        topk: §9.2's ``k`` (frozen at 3; exposed for the hand-computed tests).
        fpr_constraint: 附录 B :1603 (0.05).
        corr_min_points: §9.4 :1031–1032 (3).

    Returns:
        The ``pesn_v543_eval_report_v1`` document, with ``per_curve`` rows (one
        per curve — the paired cluster-bootstrap input of §9.4) and the
        ``pooled`` block (the §9.1 leaf table).
    """
    if not frames:
        raise V543EvalError("V543_EVAL_EMPTY", "no frames to evaluate")
    # ``frame_terms`` validates through ``EvalFrame.check`` — one pass only.
    terms = [frame_terms(frame, topk=int(topk)) for frame in frames]

    curve_ids = [frame.curve_id for frame in frames]
    ordered_curves = sorted(set(curve_ids))

    # --- per-curve rows ----------------------------------------------------
    per_curve: list[dict[str, Any]] = []
    for curve in ordered_curves:
        picked = [i for i, cid in enumerate(curve_ids) if cid == curve]
        per_curve.append(_curve_row(
            curve, [frames[i] for i in picked], [terms[i] for i in picked],
            train_q_mean=train_q_mean, corr_min_points=int(corr_min_points)))

    # --- pooled blocks -----------------------------------------------------
    move_terms = [term for term in terms if term.move_eligible]
    move_pooled = _pooled_move(move_terms, ordered_curves, per_curve)
    q_pooled = _pooled_q(per_curve, frames, train_q_mean=train_q_mean,
                         corr_min_points=int(corr_min_points))
    ordering_pooled = _pooled_ordering(frames)
    endpoint_pooled = _pooled_endpoint(frames, tau_end=tau_end,
                                       fpr_constraint=float(fpr_constraint))

    validity = {
        "q": {
            "n_eval_curves": q_pooled["n_eval_curves"],
            "median_points_per_curve": q_pooled["median_points_per_curve"],
            "frac_curves_ge3": q_pooled["frac_curves_ge3"],
            "frac_curves_ge5": q_pooled["frac_curves_ge5"],
            "corr_min_points": int(corr_min_points),
            "corr_diagnostic_only": q_pooled["corr_diagnostic_only"],
            "frac_ge3_threshold": Q_FRAC_GE3_MIN,
            "rule": (
                "plan §9.4 :1029-1033 — report n_eval_curves, "
                "median_points_per_curve, frac_curves_ge3, frac_curves_ge5; a "
                "curve needs >=3 valid q points before any correlation is "
                "computed; if frac_curves_ge3 < 0.70 then q.corr_curve_median "
                "is DIAGNOSTIC ONLY and q MAE + ordering remain the primary "
                "metrics"),
        },
        "move": {
            "n_frames": len(frames),
            "n_frames_move_eligible": int(
                sum(1 for term in terms if term.move_eligible)),
            "n_frames_excluded_by_label_status": int(
                sum(1 for term in terms if not term.move_eligible)),
            "move_eligibility_rule": (
                "frozen _frame_mask_for: only label_status == 'ok' frames "
                "carry a usable y_teacher; the other frames stay in the frame "
                "set for PROCESS but never enter a MOVE denominator"),
            "n_curves": len(ordered_curves),
            "n_curves_with_move": int(sum(
                1 for row in per_curve
                if row["move"]["rmse_zero"] is not None)),
            "n_key_rows": int(sum(term.n_key_rows for term in terms)),
            "n_frames_with_changed_pairs": int(
                sum(1 for term in terms if term.n_pairs > 0)),
            "n_pairs": int(sum(term.n_pairs for term in terms)),
            "n_active_pairs": int(sum(term.n_active_pairs for term in terms)),
            "pair_sign_threshold_angstrom": SIGN_PAIR_THRESHOLD,
            "effective_atom_threshold_angstrom": EFFECTIVE_ATOM_THRESHOLD,
            "atom_dead_threshold_angstrom": ATOM_DEAD_THRESHOLD,
        },
        "endpoint": {
            "n_rows": endpoint_pooled["n_unmasked"],
            "n_pos": endpoint_pooled["n_pos"],
            "n_neg": endpoint_pooled["n_neg"],
            "prevalence": endpoint_pooled["prevalence"],
            "tau_end": endpoint_pooled["tau_end"],
        },
        "pair_set": {
            "source": (
                "explicit pair_slots/pair_offsets per frame (the changed K-pair "
                "set); the evaluator never derives it from a model"),
            "n_pairs": int(sum(term.n_pairs for term in terms)),
        },
    }

    return {
        "schema": REPORT_SCHEMA,
        "plan_section": PLAN_SECTION,
        "evaluator_spec_sha256": evaluator_spec_sha256(),
        "tau_end": tau_end,
        "per_curve": per_curve,
        "pooled": {
            "move": move_pooled,
            "process": {
                "q": q_pooled,
                "ordering": ordering_pooled,
                "endpoint": endpoint_pooled,
            },
        },
        "validity": validity,
        "frozen_definitions": _frozen_definitions(),
        "aggregation": _aggregation_notes(),
        "bond_scale_note": dict(BOND_SCALE_NOTE),
        "ambiguities": ambiguities(),
    }


def _curve_row(curve: str, frames: Sequence[EvalFrame],
               terms: Sequence[FrameTerms], *,
               train_q_mean: float | None,
               corr_min_points: int) -> dict[str, Any]:
    """The per-curve row (one bootstrap cluster of §9.4).

    MOVE is aggregated over the ``move_eligible`` frames ONLY; q / ordering /
    endpoint use every frame of the curve (their labels exist for all §4.1
    statuses).
    """
    move_terms = [term for term in terms if term.move_eligible]
    if move_terms:
        mse_mean = float(np.mean([term.mse for term in move_terms]))
        mse_zero_mean = float(np.mean([term.mse_zero for term in move_terms]))
        rmse = math.sqrt(mse_mean)
        rmse_zero_denominator = math.sqrt(mse_zero_mean)
    else:
        mse_mean = mse_zero_mean = float("nan")
        rmse = rmse_zero_denominator = float("nan")

    n_active = int(sum(term.n_active_pairs for term in move_terms))
    n_dead = int(sum(term.n_dead_pairs for term in move_terms))
    dd_count = int(sum(term.dd_count for term in move_terms))
    dd_error = float(sum(term.dd_abs_error_sum for term in move_terms))

    cos_values = np.concatenate(
        [term.cos_values for term in move_terms if term.cos_values.size]
    ) if any(term.cos_values.size for term in move_terms) else np.zeros(0)
    basis_values = np.concatenate(
        [term.basis_residuals for term in move_terms
         if term.basis_residuals.size]
    ) if any(term.basis_residuals.size for term in move_terms) else np.zeros(0)

    fixed_values = [term.fixed_direction for term in move_terms]
    overlap_values = [term.topk_overlap for term in move_terms]
    n_fixed_frames = int(sum(1 for v in fixed_values if v is not None))

    finite_rmse = math.isfinite(rmse)
    move = {
        "rmse": rmse if finite_rmse else None,
        "rmse_zero_denominator": (rmse_zero_denominator
                                  if math.isfinite(rmse_zero_denominator)
                                  else None),
        "rmse_zero": (_ratio(rmse, rmse_zero_denominator)
                      if finite_rmse else None),
        "sign_accuracy": _mean([term.sign_accuracy for term in move_terms]),
        "n_active_pairs": n_active,
        "n_dead_pairs": n_dead,
        "dead_zone_fraction": (float(n_dead) / float(n_active)
                               if n_active else None),
        "atom_dead_fraction": (
            float(sum(term.n_dead_atoms for term in move_terms))
            / float(sum(term.n_key_rows for term in move_terms))
            if sum(term.n_key_rows for term in move_terms) else None),
        "fixed_coverage_direction_at_3": _mean(fixed_values),
        "fixed_coverage_topk_overlap_at_3": _mean(overlap_values),
        "n_fixed_coverage_frames": n_fixed_frames,
        "pair_delta_d_mae": (dd_error / float(dd_count) if dd_count else None),
        "cos_eff_median": (float(np.median(cos_values))
                           if cos_values.size else None),
        "n_effective_atoms": int(cos_values.size),
        "basis_projection_residual_median": (
            float(np.median(basis_values)) if basis_values.size else None),
        "n_basis_rows": int(basis_values.size),
        "n_frames": len(frames),
        "n_frames_move_eligible": len(move_terms),
        "n_key_rows": int(sum(term.n_key_rows for term in move_terms)),
    }

    # --- q (curve-level) ---------------------------------------------------
    q_rows = [(f.q_pred, f.q_true) for f in frames if f.q_mask]
    errors = [abs(float(pred) - float(true)) for pred, true in q_rows]
    corr = None
    # The V542 leaf computed a Pearson on any curve with >=2 non-degenerate
    # points; kept ONLY as a historical anchor (§9.4 :1033).
    corr_v542 = None
    if len(q_rows) >= 2:
        corr_v542 = _pearson(np.asarray([float(p) for p, _ in q_rows]),
                             np.asarray([float(t) for _, t in q_rows]))
    if len(q_rows) >= int(corr_min_points):
        corr = corr_v542
    constant_mae = (
        float(np.mean([abs(float(train_q_mean) - float(true))
                       for _, true in q_rows]))
        if (q_rows and train_q_mean is not None) else None)
    q_block = {
        "mae": (float(np.mean(errors)) if errors else None),
        "n_points": len(q_rows),
        "corr": corr,
        "corr_n_points": len(q_rows),
        "corr_v542": corr_v542,
        "constant_q_baseline_mae": constant_mae,
    }

    # --- ordering (curve-level, V542 (curve, direction) groups) ------------
    ordering_pair_accuracy = _frozen_ordering()
    q_pred = [float(f.q_pred) for f in frames if f.q_mask]
    q_true = [float(f.q_true) for f in frames if f.q_mask]
    groups = [(f.curve_id, f.direction) for f in frames if f.q_mask]
    full = ordering_pair_accuracy(q_pred, q_true, groups, gap=0.0,
                                  cross_ts_only=False)
    cross = ordering_pair_accuracy(q_pred, q_true, groups, gap=0.0,
                                   cross_ts_only=True)
    ordering = {
        "pairwise_accuracy_full": full["accuracy"],
        "n_pairs_full": int(full["n_pairs"]),
        "pairwise_accuracy_cross_ts": cross["accuracy"],
        "n_pairs_cross_ts": int(cross["n_pairs"]),
    }

    # --- endpoint (curve-level) -------------------------------------------
    endpoint_rows = [(float(f.endpoint_logit), int(f.endpoint_label))
                     for f in frames if f.endpoint_mask]
    endpoint = _endpoint_block(
        [row[0] for row in endpoint_rows], [row[1] for row in endpoint_rows])
    endpoint["n_rows"] = len(endpoint_rows)

    return {
        "curve_id": str(curve),
        "n_frames": len(frames),
        "n_key_rows": move["n_key_rows"],
        "move": move,
        "q": q_block,
        "ordering": ordering,
        "endpoint": endpoint,
    }


def _pooled_move(terms: Sequence[FrameTerms], curves: Sequence[str],
                 per_curve: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The pooled MOVE block (the §9.1 leaf table, V542 aggregation)."""
    def curve_values(key: str, nested: str = "move") -> list[float | None]:
        return [row[nested].get(key) for row in per_curve]

    rmse_zero_values = curve_values("rmse_zero")
    sign_values = curve_values("sign_accuracy")
    fixed_values = curve_values("fixed_coverage_direction_at_3")
    overlap_values = curve_values("fixed_coverage_topk_overlap_at_3")
    dead_values = curve_values("dead_zone_fraction")
    atom_dead_values = curve_values("atom_dead_fraction")
    dd_values = curve_values("pair_delta_d_mae")
    cos_values = curve_values("cos_eff_median")
    basis_values = curve_values("basis_projection_residual_median")

    n_active = int(sum(term.n_active_pairs for term in terms))
    n_dead = int(sum(term.n_dead_pairs for term in terms))
    n_keys = int(sum(term.n_key_rows for term in terms))
    n_dead_atoms = int(sum(term.n_dead_atoms for term in terms))
    dd_count = int(sum(term.dd_count for term in terms))
    dd_error = float(sum(term.dd_abs_error_sum for term in terms))

    cos_all = np.concatenate(
        [term.cos_values for term in terms if term.cos_values.size]
    ) if any(term.cos_values.size for term in terms) else np.zeros(0)
    basis_all = np.concatenate(
        [term.basis_residuals for term in terms if term.basis_residuals.size]
    ) if any(term.basis_residuals.size for term in terms) else np.zeros(0)

    fixed_frames = [(term.fixed_direction, term.stratum, term.intent_class)
                    for term in terms if term.fixed_direction is not None]
    frame_mean_fixed = _mean([v for v, _, _ in fixed_frames])
    frame_mean_overlap = _mean(
        [term.topk_overlap for term in terms if term.topk_overlap is not None])

    rmse = _median([row["move"]["rmse"] for row in per_curve])
    rmse_zero_denominator = _median(
        [row["move"]["rmse_zero_denominator"] for row in per_curve])

    return {
        "rmse": rmse,
        "rmse_zero_denominator": rmse_zero_denominator,
        "rmse_zero": _median(rmse_zero_values),
        "rmse_zero_mean": _mean(rmse_zero_values),
        "rmse_zero_aggregation": (
            "per-curve sqrt(mean frame mse)/sqrt(mean frame mse_zero); pooled "
            "= MEDIAN over curves (the frozen V542 "
            "key_rmse_over_zero.median_rmse_over_zero leaf)"),
        "sign_accuracy_cluster_median": _median(sign_values),
        "sign_accuracy_cluster_mean": _mean(sign_values),
        "sign_accuracy_aggregation": (
            "per-frame G5.3 sign accuracy on |Δd_y| >= 0.005 Å pairs (a "
            "predicted |Δd_u| < 0.005 Å counts INCORRECT); per-curve frame "
            "mean; pooled = median over curves"),
        "dead_zone_fraction": (float(n_dead) / float(n_active)
                               if n_active else None),
        "dead_zone_fraction_curve_median": _median(dead_values),
        "dead_fraction": (float(n_dead) / float(n_active)
                          if n_active else None),
        "dead_fraction_definition": (
            "H2 uses the frozen V542 leaf sre_and_sign.dead_zone_fraction "
            "(teacher-active pairs whose predicted |Δd| < 0.005 Å); the plan "
            "§9.1 :878 wording is reported as atom_dead_fraction"),
        "atom_dead_fraction": (float(n_dead_atoms) / float(n_keys)
                               if n_keys else None),
        "atom_dead_fraction_curve_median": _median(atom_dead_values),
        "atom_dead_threshold_angstrom": ATOM_DEAD_THRESHOLD,
        "fixed_coverage": {
            "direction_at_3": _mean(fixed_values),
            "direction_at_3_curve_median": _median(fixed_values),
            "direction_at_3_frame_mean": frame_mean_fixed,
            "topk_overlap_at_3": _mean(overlap_values),
            "topk_overlap_at_3_curve_median": _median(overlap_values),
            "topk_overlap_at_3_frame_mean": frame_mean_overlap,
            "k": int(FIXED_COVERAGE_TOPK),
            "pair_sign_threshold_angstrom": SIGN_PAIR_THRESHOLD,
            "n_frames": len(fixed_frames),
            "n_curves": int(sum(1 for v in fixed_values if v is not None)),
            "aggregation": FIXED_COVERAGE_AGGREGATION,
            "definition": (
                "plan §9.2 :904-907 — T_k = teacher top-k |Δd_y| pairs (ties "
                "by pair index); direction@3 = mean over l in T_k of "
                "[sign(Δd_u[l])==sign(Δd_y[l]) AND |Δd_u[l]|>=0.005 Å]; "
                "topk_overlap@3 = |P_k ∩ T_k|/k with P_k the predicted |Δd_u| "
                "top-k"),
            "by_stratum": _grouped_fixed(fixed_frames, index=1),
            "by_intent_class": _grouped_fixed(fixed_frames, index=2),
        },
        "fixed_coverage_direction_at_3": _mean(fixed_values),
        "fixed_coverage_topk_overlap_at_3": _mean(overlap_values),
        "pair_delta_d_mae": (dd_error / float(dd_count) if dd_count else None),
        "pair_delta_d_mae_curve_median": _median(dd_values),
        "pair_delta_d_definition": (
            "plan §9.1 :881 — mean |Δd_u − Δd_y| over the changed K-pairs (Å)"),
        "cos_eff_median": (float(np.median(cos_all)) if cos_all.size else None),
        "cos_eff_median_curve_median": _median(cos_values),
        "n_effective_atoms": int(cos_all.size),
        "n_effective_atoms_threshold_angstrom": EFFECTIVE_ATOM_THRESHOLD,
        "basis_projection_residual_median": (
            float(np.median(basis_all)) if basis_all.size else None),
        "basis_projection_residual_curve_median": _median(basis_values),
        "n_basis_rows": int(basis_all.size),
        "n_curves": len(curves),
        "n_frames": len(terms),
    }


def _grouped_fixed(rows: Sequence[tuple[float | None, str, str]], *,
                   index: int) -> dict[str, Any]:
    """Frame-level fixed-coverage means per stratum / per reaction class."""
    buckets: dict[str, list[float]] = {}
    for row in rows:
        value = row[0]
        label = row[index]
        if value is None:
            continue
        buckets.setdefault(str(label), []).append(float(value))
    return {label: {"n_frames": len(values),
                    "direction_at_3_frame_mean": float(np.mean(values))}
            for label, values in sorted(buckets.items())}


def _pooled_q(per_curve: Sequence[Mapping[str, Any]],
              frames: Sequence[EvalFrame], *,
              train_q_mean: float | None, corr_min_points: int
              ) -> dict[str, Any]:
    """The pooled q block (§9.1 :887–891 + the §9.4 validity rule)."""
    curve_mae = [row["q"]["mae"] for row in per_curve
                 if row["q"]["mae"] is not None]
    points = [int(row["q"]["n_points"]) for row in per_curve
              if row["q"]["n_points"] > 0]
    corr_values = [row["q"]["corr"] for row in per_curve
                   if row["q"]["corr"] is not None]
    corr_v542_values = [row["q"]["corr_v542"] for row in per_curve
                        if row["q"].get("corr_v542") is not None]
    n_curves = len(points)
    frac_ge3 = (float(sum(1 for n in points if n >= Q_CORR_MIN_POINTS))
                / float(n_curves)) if n_curves else None
    frac_ge5 = (float(sum(1 for n in points if n >= Q_GE5_MIN_POINTS))
                / float(n_curves)) if n_curves else None
    q_rows = [(float(f.q_true)) for f in frames if f.q_mask]
    const_mae = (float(np.mean([abs(float(train_q_mean) - true)
                                for true in q_rows]))
                 if (q_rows and train_q_mean is not None) else None)
    median_mae = _median(curve_mae)
    diagnostic_only = (frac_ge3 is None
                       or float(frac_ge3) < Q_FRAC_GE3_MIN)
    return {
        "mae_curve_median": median_mae,
        "mae_curve_mean": _mean(curve_mae),
        "n_eval_curves": n_curves,
        "median_points_per_curve": _median([float(n) for n in points]),
        "total_points": int(sum(points)),
        "frac_curves_ge3": frac_ge3,
        "frac_curves_ge5": frac_ge5,
        "corr_min_points": int(corr_min_points),
        "corr_curve_median": _median(corr_values),
        "corr_curve_mean": _mean(corr_values),
        "corr_curve_median_v542_anchor": _median(corr_v542_values),
        "n_curves_corr": len(corr_values),
        "corr_definition": (
            "Pearson over a curve's masked q points, computed ONLY when the "
            "curve has >= "
            f"{int(corr_min_points)} points (plan §9.4 :1031); "
            "corr_curve_median_v542_anchor keeps the V542 any-2-point reading "
            "as a historical anchor and is never a conclusion (§9.4 :1033)"),
        "corr_diagnostic_only": bool(diagnostic_only),
        "corr_diagnostic_rule": (
            "plan §9.4 :1032 — frac_curves_ge3 < 0.70 => q.corr_curve_median "
            "is diagnostic only; q MAE and ordering remain the primary "
            "metrics"),
        "constant_q_baseline_mae": const_mae,
        "constant_q_value": train_q_mean,
        "learnable_above_constant": (
            None if (const_mae is None or median_mae is None)
            else bool(median_mae < const_mae)),
        "aggregation": (
            "q MAE per curve = mean |q_pred − q_true| over the curve's masked "
            "frames; pooled = median (primary) and mean over curves"),
    }


def _pooled_ordering(frames: Sequence[EvalFrame]) -> dict[str, Any]:
    """The pooled ordering block — the frozen V542 pair counting."""
    ordering_pair_accuracy = _frozen_ordering()
    q_pred = [float(f.q_pred) for f in frames if f.q_mask]
    q_true = [float(f.q_true) for f in frames if f.q_mask]
    groups = [(f.curve_id, f.direction) for f in frames if f.q_mask]
    full = ordering_pair_accuracy(q_pred, q_true, groups, gap=0.0,
                                  cross_ts_only=False)
    cross = ordering_pair_accuracy(q_pred, q_true, groups, gap=0.0,
                                   cross_ts_only=True)
    return {
        "pairwise_accuracy_full": full["accuracy"],
        "n_pairs_full": int(full["n_pairs"]),
        "pairwise_accuracy_cross_ts": cross["accuracy"],
        "n_pairs_cross_ts": int(cross["n_pairs"]),
        "source": (
            "frozen V542 ordering_pair_accuracy over (curve, intent "
            "direction) groups; cross_ts keeps only pairs with strictly "
            "opposite teacher sides"),
    }


def _pooled_endpoint(frames: Sequence[EvalFrame], *,
                     tau_end: float | None,
                     fpr_constraint: float) -> dict[str, Any]:
    """The pooled endpoint block (§9.1 :883–886, ``tau_end=None`` §1.5)."""
    rows = [(float(f.endpoint_logit), int(f.endpoint_label), f.is_start_row,
             f.is_ts_row) for f in frames if f.endpoint_mask]
    keep = [(logit, label) for logit, label, _s, _t in rows
            if int(label) in (0, 1)]
    average_precision, auc_mann_whitney, _sigmoid = _frozen_endpoint_math()
    logits = [row[0] for row in keep]
    labels = [int(row[1]) for row in keep]
    scores = [_sigmoid_scalar(v) for v in logits]
    y = np.asarray(labels, dtype=np.int64)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    auprc = (average_precision(scores, labels)
             if (n_pos and n_neg) else None)
    auroc = (auc_mann_whitney(scores, labels)
             if (n_pos and n_neg) else None)
    operating = recall_at_fpr(scores, labels, fpr_constraint=fpr_constraint)
    n_start = int(sum(1 for _l, _y, start, _t in rows if start))
    n_ts = int(sum(1 for _l, _y, _s, ts in rows if ts))
    mis_start = None
    mis_ts = None
    if tau_end is not None and rows:
        start_scores = [(_sigmoid_scalar(logit) >= float(tau_end))
                        for logit, _y, start, _t in rows if start]
        ts_scores = [(_sigmoid_scalar(logit) >= float(tau_end))
                     for logit, _y, _s, ts in rows if ts]
        if start_scores:
            mis_start = float(np.mean(start_scores))
        if ts_scores:
            mis_ts = float(np.mean(ts_scores))
    brier = (float(np.mean((np.asarray(scores) - y) ** 2))
             if y.size else None)
    return {
        "auprc": auprc,
        "auroc": auroc,
        "brier": brier,
        "ece": (_ece(np.asarray(scores, dtype=np.float64), y)
                if y.size else None),
        "ece_bins": int(ECE_BINS),
        "recall_at_fpr_0p05": operating["recall"],
        "fpr_at_threshold": operating["fpr"],
        "threshold_at_fpr_constraint": operating["threshold"],
        "fpr_constraint": float(fpr_constraint),
        "operating_point_rule": operating["rule"],
        "prevalence": (float(y.mean()) if y.size else None),
        "n_unmasked": int(y.size),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "tau_end": tau_end,
        "start_misstop_rate_at_tau": mis_start,
        "ts_misstop_rate_at_tau": mis_ts,
        "n_start_rows": n_start,
        "n_ts_rows": n_ts,
        "tau_note": (
            "tau_end=None is the frozen §1.5 :105 V542 semantics: no threshold "
            "is calibrated by this evaluator and every tau-dependent field is "
            "None. §9.5 :1040 calibrates tau ONCE on inner-val (target "
            "precision 0.9) outside this module"),
    }


# ---------------------------------------------------------------------------
# the zero-residual baseline (M0 / P0)
# ---------------------------------------------------------------------------
def zero_residual_frames(frames: Sequence[EvalFrame]) -> list[EvalFrame]:
    """M0/P0 of §4.3 :226 — the same frames with the zero residual.

    ``u = u0``, ``q = tanh(q0_raw)``, ``endpoint_logit = endpoint0_logit``.
    The frame set, the pair terms and every teacher value are IDENTICAL to the
    candidate's, which is what makes the H2/H3 pairing (§9.4 :1019) valid.
    """
    out: list[EvalFrame] = []
    for frame in frames:
        frame.check()
        if frame.u0 is None or frame.q0_raw is None or (
                frame.endpoint0_logit is None):
            raise V543EvalError(
                "V543_EVAL_BASELINE_INPUT",
                f"curve {frame.curve_id!r}: the zero-residual baseline needs "
                "u0, q0_raw and endpoint0_logit on every frame")
        out.append(EvalFrame(
            curve_id=frame.curve_id,
            u=np.asarray(frame.u0, dtype=np.float64),
            y=np.asarray(frame.y, dtype=np.float64),
            pair_slots=frame.pair_slots,
            pair_offsets=frame.pair_offsets,
            direction=frame.direction,
            stratum=frame.stratum,
            intent_class=frame.intent_class,
            q_pred=math.tanh(float(frame.q0_raw)),
            q_true=frame.q_true,
            q_mask=frame.q_mask,
            endpoint_logit=float(frame.endpoint0_logit),
            endpoint_label=frame.endpoint_label,
            endpoint_mask=frame.endpoint_mask,
            is_start_row=frame.is_start_row,
            is_ts_row=frame.is_ts_row,
            move_eligible=frame.move_eligible,
            u0=frame.u0,
            q0_raw=frame.q0_raw,
            endpoint0_logit=frame.endpoint0_logit,
            key_v=frame.key_v,
        ))
    return out


def evaluate_baseline(frames: Sequence[EvalFrame], **kwargs: Any) -> dict[str, Any]:
    """Evaluate the M0/P0 zero-residual baseline on the SAME frame set.

    This is the §9.5 :1037 clause ("S0 在 V543 evaluator 下重算 M0/P0") in code:
    the baseline goes through exactly the same metric code as a candidate, so
    the only difference between the two report documents is the prediction.
    """
    report = evaluate_v543(zero_residual_frames(frames), **kwargs)
    report["baseline"] = {
        "kind": "M0P0_zero_residual",
        "definition": (
            "plan §4.3 :226 — inactive refiners are the zero residual: "
            "u = u0 (M0), q = tanh(q0_raw) and endpoint_logit = e0 (P0)"),
        "plan_lines": "§4.3 :226 / §9.5 :1037",
    }
    return report


# ---------------------------------------------------------------------------
# paired curve-level cluster bootstrap (plan §9.4; consumed by v543_stats)
# ---------------------------------------------------------------------------
def paired_curve_bootstrap(
        candidate: Mapping[str, float] | Sequence[Mapping[str, float]],
        baseline: Mapping[str, float] | Sequence[Mapping[str, float]],
        *,
        higher_is_better: bool,
        metric: str = "",
        draws: int = BOOTSTRAP_DRAWS,
        seed_root: int = BOOTSTRAP_SEED_ROOT,
) -> dict[str, Any]:
    """Paired curve-cluster bootstrap of one metric (plan §9.3/§9.4).

    The inputs are ``curve_id -> value`` tables (one per seed, or a single
    table).  Implements §9.4 literally:

    * unit = curve (reaction cluster) — never a frame (:1025);
    * pairing = same cluster, same index for candidate and baseline (:1019);
    * missing clusters dropped on BOTH sides, count reported, never imputed
      (:1024);
    * per draw the seeds are averaged FIRST, then compared (:1023);
    * percentile 95 % CI (:1021) and the one-sided improvement p-value
      ``(1 + #{draw <= 0}) / (draws + 1)`` (:1022 / §9.3 :1000).

    ``higher_is_better`` selects the improvement sign (direction / AUPRC / sign
    accuracy up is good; RMSE / MAE down is good).

    .. note:: **this is the SHARED-INTERSECTION mode, kept as the audit /
       compatibility path.**  It resamples ONE curve set (the intersection over
       every seed and both sides) and averages the seeds inside each draw.  That
       reads §9.4 :1019 ("pairing: same cluster, same frame set") as a
       requirement on the CLUSTER SET rather than on the pairing, which is not
       what the statistics lane needs: each formal seed evaluates its OWN
       inner-val curve sample, so the all-seed intersection collapses (measured:
       37 curves out of ~1,020 per seed), which destroys the power of the
       10,000-draw confirmatory test.  For the confirmatory H3 statistic use
       :func:`paired_curve_bootstrap_multiseed`; both modes are reported side by
       side and this one is still the audit reference (identical seed sets must
       agree with it).
    """
    cand_seeds = _as_seed_tables(candidate)
    base_seeds = _as_seed_tables(baseline)
    if len(cand_seeds) != len(base_seeds):
        raise V543EvalError(
            "V543_EVAL_BOOTSTRAP_SEEDS",
            f"candidate has {len(cand_seeds)} seed tables, baseline "
            f"{len(base_seeds)}")
    shared = sorted(set.intersection(
        *(set(table) for table in (*cand_seeds, *base_seeds))))
    dropped_candidate = sorted(
        set().union(*(set(table) for table in cand_seeds)) - set(shared))
    dropped_baseline = sorted(
        set().union(*(set(table) for table in base_seeds)) - set(shared))
    if not shared:
        raise V543EvalError(
            "V543_EVAL_BOOTSTRAP_EMPTY",
            "candidate and baseline share no curve; nothing can be paired")
    if int(draws) <= 0:
        raise V543EvalError("V543_EVAL_BOOTSTRAP_DRAWS",
                            f"draws must be positive, got {draws}")

    improvement = np.empty((len(cand_seeds), len(shared)), dtype=np.float64)
    for s, (cand, base) in enumerate(zip(cand_seeds, base_seeds)):
        for c, curve in enumerate(shared):
            c_value = float(cand[curve])
            b_value = float(base[curve])
            if not (math.isfinite(c_value) and math.isfinite(b_value)):
                raise V543EvalError(
                    "V543_EVAL_BOOTSTRAP_NONFINITE",
                    f"metric {metric!r} curve {curve!r} seed index {s}: "
                    f"candidate {c_value} / baseline {b_value}")
            improvement[s, c] = (c_value - b_value) if higher_is_better \
                else (b_value - c_value)

    rng = np.random.default_rng(int(seed_root))
    n = int(len(shared))
    draw_values = np.empty(int(draws), dtype=np.float64)
    for d in range(int(draws)):
        pick = rng.integers(0, n, size=n)
        # §9.4 :1023 — seed-average first, compare afterwards.
        draw_values[d] = float(np.mean(improvement[:, pick]))

    point = float(np.mean(improvement))
    lower = float(np.percentile(draw_values, CI_LOW_PERCENTILE))
    upper = float(np.percentile(draw_values, CI_HIGH_PERCENTILE))
    n_nonpositive = int((draw_values <= 0.0).sum())
    p_value = float((1 + n_nonpositive) / (int(draws) + 1))
    return {
        "metric": str(metric),
        "higher_is_better": bool(higher_is_better),
        "unit": BOOTSTRAP_UNIT,
        "pairing": BOOTSTRAP_PAIRING,
        "seed_aggregation": BOOTSTRAP_SEED_AGGREGATION,
        "n_curves": n,
        "n_seeds": len(cand_seeds),
        "draws": int(draws),
        "seed_root": int(seed_root),
        "point_estimate": point,
        "ci95": [lower, upper],
        "ci_percentiles": [CI_LOW_PERCENTILE, CI_HIGH_PERCENTILE],
        "n_draws_nonpositive": n_nonpositive,
        "p_one_sided": p_value,
        "p_rule": "(1 + #{seed-averaged paired improvement <= 0}) / (draws + 1)",
        "dropped_curves": {
            "n_candidate_only": len(dropped_candidate),
            "n_baseline_only": len(dropped_baseline),
            "candidate_only": dropped_candidate[:50],
            "baseline_only": dropped_baseline[:50],
            "note": ("plan §9.4 :1024 — missing clusters are dropped on BOTH "
                     "sides and reported; never imputed"),
        },
    }


def _as_seed_tables(value: Mapping[str, float]
                    | Sequence[Mapping[str, float]]
                    ) -> list[Mapping[str, float]]:
    if isinstance(value, Mapping):
        return [value]
    tables = list(value)
    if not tables or not all(isinstance(t, Mapping) for t in tables):
        raise V543EvalError(
            "V543_EVAL_BOOTSTRAP_INPUT",
            "bootstrap input must be one curve->value mapping or a non-empty "
            "sequence of them (one per seed)")
    return [dict(t) for t in tables]


#: Mode id of the confirmatory (per-seed resampling) bootstrap.
BOOTSTRAP_MODE_MULTISEED: Final[str] = "per_seed_independent"
#: Mode id of the audit/compatibility (shared intersection) bootstrap.
BOOTSTRAP_MODE_SHARED: Final[str] = "shared_intersection"

#: How the per-draw / per-seed RNG streams are derived.  Recorded so a reader —
#: or a test — can reproduce every draw without reading the source.
BOOTSTRAP_SEED_DERIVATION: Final[str] = (
    "np.random.default_rng([seed_root, draw_index, seed_index]) — one "
    "independent stream per (draw, seed); candidate and baseline of the same "
    "seed share that seed's resample indices")


def paired_curve_bootstrap_multiseed(
        candidate: Mapping[str, float] | Sequence[Mapping[str, float]],
        baseline: Mapping[str, float] | Sequence[Mapping[str, float]],
        *,
        higher_is_better: bool,
        metric: str = "",
        draws: int = BOOTSTRAP_DRAWS,
        seed_root: int = BOOTSTRAP_SEED_ROOT,
        keep_draws: bool = True,
) -> dict[str, Any]:
    """Confirmatory multi-seed paired curve-cluster bootstrap (plan §9.3/§9.4).

    Why this mode exists
    --------------------
    Each formal seed evaluates ITS OWN inner-val curve sample.  Reading §9.4
    :1019's "pairing = same cluster, same frame set" as a requirement that the
    three seeds share one cluster set collapses the sample to the all-seed
    intersection (measured on this campaign: 37 curves out of ~1,020 per seed),
    which cannot support the 10,000-draw H3 test.  The clause is a PAIRING
    discipline inside a comparison (candidate and baseline resample the SAME
    cluster indices); :1023 ("每 draw 先 seed-average，再比较") already
    presupposes that each seed contributes its own statistic.

    Therefore, per draw ``d``:

    * each seed ``s`` resamples ITS OWN curve set with an independent stream
      derived from ``seed_root`` (:data:`BOOTSTRAP_SEED_DERIVATION`);
    * the candidate and the baseline of seed ``s`` share that seed's resample
      indices — the pairing of :1019, cluster by cluster;
    * seed ``s``'s statistic is the mean paired improvement over its resampled
      clusters;
    * the draw value is the mean of those per-seed statistics (:1023's
      seed-average, applied per draw).

    The 10,000 draw values then give the percentile 95 % CI (:1021) and the
    one-sided p ``(1 + #{draw <= 0}) / (draws + 1)`` (:1022 / §9.3 :1000).

    Missing clusters are still dropped per seed on BOTH sides of that seed's
    comparison and counted (never imputed, :1024); the counts are reported per
    seed and totalled.  No seed may end up with an empty comparison — that is a
    typed refusal rather than a silently dropped seed.

    Args:
        candidate/baseline: one ``curve_id -> value`` mapping per seed (or a
            single mapping for a one-seed check).  Seed order defines the
            ``seed_index`` used by :data:`BOOTSTRAP_SEED_DERIVATION`, so pass the
            campaign's seed order (``[17, 29, 43]``).
        keep_draws: include the reproducible per-draw values in the result.

    Returns:
        A result block whose ``mode`` is :data:`BOOTSTRAP_MODE_MULTISEED`, with
        per-seed curve counts / dropped counts / point estimates next to the
        pooled CI and p.
    """
    cand_seeds = _as_seed_tables(candidate)
    base_seeds = _as_seed_tables(baseline)
    if len(cand_seeds) != len(base_seeds):
        raise V543EvalError(
            "V543_EVAL_BOOTSTRAP_SEEDS",
            f"candidate has {len(cand_seeds)} seed tables, baseline "
            f"{len(base_seeds)}")
    if int(draws) <= 0:
        raise V543EvalError("V543_EVAL_BOOTSTRAP_DRAWS",
                            f"draws must be positive, got {draws}")

    improvements: list[np.ndarray] = []
    per_seed: list[dict[str, Any]] = []
    total_dropped_candidate = 0
    total_dropped_baseline = 0
    for index, (cand, base) in enumerate(zip(cand_seeds, base_seeds)):
        shared = sorted(set(cand) & set(base))
        dropped_candidate = sorted(set(cand) - set(shared))
        dropped_baseline = sorted(set(base) - set(shared))
        if not shared:
            raise V543EvalError(
                "V543_EVAL_BOOTSTRAP_EMPTY",
                f"seed index {index} shares no curve between candidate and "
                f"baseline ({len(cand)} vs {len(base)} curves); the per-draw "
                "seed-average has nothing to average for that seed")
        improvement = np.empty(len(shared), dtype=np.float64)
        for position, curve in enumerate(shared):
            c_value = float(cand[curve])
            b_value = float(base[curve])
            if not (math.isfinite(c_value) and math.isfinite(b_value)):
                raise V543EvalError(
                    "V543_EVAL_BOOTSTRAP_NONFINITE",
                    f"metric {metric!r} curve {curve!r} seed index {index}: "
                    f"candidate {c_value} / baseline {b_value}")
            improvement[position] = (c_value - b_value) if higher_is_better \
                else (b_value - c_value)
        improvements.append(improvement)
        total_dropped_candidate += len(dropped_candidate)
        total_dropped_baseline += len(dropped_baseline)
        per_seed.append({
            "seed_index": int(index),
            "n_curves": int(improvement.size),
            "point_estimate": float(improvement.mean()),
            "dropped_curves": {
                "n_candidate_only": len(dropped_candidate),
                "n_baseline_only": len(dropped_baseline),
                "candidate_only": dropped_candidate[:50],
                "baseline_only": dropped_baseline[:50],
            },
        })

    draw_values = np.empty(int(draws), dtype=np.float64)
    for d in range(int(draws)):
        statistics = np.empty(len(improvements), dtype=np.float64)
        for index, improvement in enumerate(improvements):
            # one independent stream per (draw, seed): the seeds do NOT share
            # clusters, but candidate and baseline of a seed DO share indices.
            generator = np.random.default_rng(
                [int(seed_root), int(d), int(index)])
            size = int(improvement.size)
            pick = generator.integers(0, size, size=size)
            statistics[index] = float(improvement[pick].mean())
        draw_values[d] = float(statistics.mean())

    point = float(np.mean([float(table["point_estimate"])
                           for table in per_seed]))
    lower = float(np.percentile(draw_values, CI_LOW_PERCENTILE))
    upper = float(np.percentile(draw_values, CI_HIGH_PERCENTILE))
    n_nonpositive = int((draw_values <= 0.0).sum())
    p_value = float((1 + n_nonpositive) / (int(draws) + 1))
    curve_counts = [int(table["n_curves"]) for table in per_seed]
    result: dict[str, Any] = {
        "mode": BOOTSTRAP_MODE_MULTISEED,
        "metric": str(metric),
        "higher_is_better": bool(higher_is_better),
        "unit": BOOTSTRAP_UNIT,
        "pairing": (
            "within a seed: candidate and baseline resample the SAME cluster "
            "indices (§9.4 :1019); across seeds: each seed resamples its OWN "
            "inner-val cluster set"),
        "seed_aggregation": BOOTSTRAP_SEED_AGGREGATION,
        "seed_derivation": BOOTSTRAP_SEED_DERIVATION,
        "n_seeds": len(cand_seeds),
        "n_curves": int(curve_counts[0]) if len(set(curve_counts)) == 1
        else int(min(curve_counts)),
        "n_curves_per_seed": curve_counts,
        "n_curves_min": int(min(curve_counts)),
        "n_curves_median": float(np.median(curve_counts)),
        "n_curves_max": int(max(curve_counts)),
        "n_curves_total": int(sum(curve_counts)),
        "draws": int(draws),
        "seed_root": int(seed_root),
        "point_estimate": point,
        "ci95": [lower, upper],
        "ci_percentiles": [CI_LOW_PERCENTILE, CI_HIGH_PERCENTILE],
        "n_draws_nonpositive": n_nonpositive,
        "p_one_sided": p_value,
        "p_rule": "(1 + #{seed-averaged paired improvement <= 0}) / (draws + 1)",
        "per_seed": per_seed,
        "dropped_curves": {
            "n_candidate_only": total_dropped_candidate,
            "n_baseline_only": total_dropped_baseline,
            "per_seed": [dict(table["dropped_curves"]) for table in per_seed],
            "note": ("plan §9.4 :1024 — missing clusters are dropped on BOTH "
                     "sides of that seed's comparison and reported per seed; "
                     "never imputed, and no seed is silently skipped"),
        },
        "why_not_one_shared_cluster_set": (
            "each formal seed evaluates its own inner-val curve sample; the "
            "all-seed intersection of ~1,020-curve samples collapses to tens of "
            "curves, which cannot support the 10,000-draw H3 test. §9.4 :1019 "
            "governs the PAIRING inside a comparison, §9.4 :1023 the per-draw "
            "seed-average"),
    }
    if keep_draws:
        result["draw_values"] = [float(v) for v in draw_values]
    return result


# ---------------------------------------------------------------------------
# identity / provenance
# ---------------------------------------------------------------------------
def evaluator_spec() -> dict[str, Any]:
    """The frozen formula/constant block — the identity of this evaluator.

    A change to ANY metric definition must change this document (and therefore
    :func:`evaluator_spec_sha256`), which is what makes a frozen baseline
    report verifiable.
    """
    return {
        "schema": REPORT_SCHEMA,
        "plan_section": PLAN_SECTION,
        "metrics": {
            "move.rmse_zero": (
                "per-curve sqrt(mean_f mean_k‖u−y‖²)/sqrt(mean_f mean_k‖y‖²); "
                "pooled = median over curves (V542 leaf); undefined "
                f"denominator < {DENOM_EPS} Å is None"),
            "move.sign_accuracy": (
                "per frame mean over teacher-significant pairs (|Δd_y| >= "
                f"{SIGN_PAIR_THRESHOLD} Å) of [sign(Δd_u)==sign(Δd_y) AND "
                f"|Δd_u| >= {SIGN_PAIR_THRESHOLD} Å]; per-curve frame mean; "
                "pooled = median over curves (V542 leaf)"),
            "move.dead_zone_fraction": (
                "teacher-active pairs whose |Δd_u| < "
                f"{SIGN_PAIR_THRESHOLD} Å, global ratio (V542 leaf)"),
            "move.atom_dead_fraction": (
                f"share of key rows with ‖u_k‖ < {ATOM_DEAD_THRESHOLD} Å"),
            "move.fixed_coverage.direction_at_3": (
                "plan §9.2 :904-906 — k = min(3,n_pairs); T_k = teacher top-k "
                "|Δd_y| (ties by pair index); mean over l in T_k of "
                "[sign(Δd_u[l])==sign(Δd_y[l]) AND |Δd_u[l]| >= "
                f"{SIGN_PAIR_THRESHOLD} Å]; frame mean -> curve mean -> seed "
                "mean"),
            "move.fixed_coverage.topk_overlap_at_3": (
                "plan §9.2 :907 — |P_k ∩ T_k| / k"),
            "move.pair_delta_d.mae": "mean |Δd_u − Δd_y| over changed K-pairs",
            "move.basis_projection_residual": (
                "median over key rows of ‖r_perp‖/‖r‖, r = y − u0 projected "
                "on span(key_v) (minimum-norm least squares)"),
            "move.cos_eff_median": (
                "G5.2 soft-norm cosine median over atoms with ‖y_k‖ > "
                f"{EFFECTIVE_ATOM_THRESHOLD} Å"),
            "process.q.mae_curve_median": (
                "per-curve mean |q_pred−q_true| over masked frames; pooled "
                "median (primary) and mean"),
            "process.q.corr_curve_median": (
                "per-curve Pearson over masked frames with >= "
                f"{Q_CORR_MIN_POINTS} points (plan §9.4 :1031); diagnostic "
                "only when frac_curves_ge3 < "
                f"{Q_FRAC_GE3_MIN}"),
            "process.ordering": (
                "frozen V542 ordering_pair_accuracy on (curve, direction) "
                "groups, gap=0.0"),
            "process.endpoint.auprc/auroc": (
                "frozen V542 average_precision / auc_mann_whitney on sigmoid "
                "endpoint logits over unmasked rows"),
            "process.endpoint.brier": "mean (p − y)²",
            "process.endpoint.ece": f"{ECE_BINS} equal-width bins over [0,1]",
            "process.endpoint.recall_at_fpr_0p05": (
                "max recall subject to achieved FPR <= "
                f"{FPR_CONSTRAINT} on the same rows; ties -> higher threshold"),
        },
        "constants": {
            "fixed_coverage_topk": int(FIXED_COVERAGE_TOPK),
            "sign_pair_threshold_angstrom": SIGN_PAIR_THRESHOLD,
            "effective_atom_threshold_angstrom": EFFECTIVE_ATOM_THRESHOLD,
            "cosine_soft_norm": COSINE_SOFT_NORM,
            "sre_min_total_change_angstrom": SRE_MIN_TOTAL_CHANGE,
            "fpr_constraint": float(FPR_CONSTRAINT),
            "ece_bins": int(ECE_BINS),
            "q_corr_min_points": int(Q_CORR_MIN_POINTS),
            "q_frac_ge3_min": float(Q_FRAC_GE3_MIN),
            "q_ge5_min_points": int(Q_GE5_MIN_POINTS),
            "atom_dead_threshold_angstrom": ATOM_DEAD_THRESHOLD,
            "denominator_epsilon_angstrom": DENOM_EPS,
            "residual_norm_floor_angstrom": RESIDUAL_NORM_FLOOR,
            "bootstrap_draws": int(BOOTSTRAP_DRAWS),
            "bootstrap_seed_root": int(BOOTSTRAP_SEED_ROOT),
            "ci_percentiles": [CI_LOW_PERCENTILE, CI_HIGH_PERCENTILE],
        },
        "aggregation": _aggregation_notes(),
        "bond_scale_note": dict(BOND_SCALE_NOTE),
    }


def evaluator_spec_sha256() -> str:
    """Canonical SHA-256 of :func:`evaluator_spec` (the frozen identity)."""
    from pes2ts_core.pesn.models.v542.identity import canonical_sha256

    return canonical_sha256(evaluator_spec())


def _frozen_definitions() -> dict[str, Any]:
    """The imported-not-copied provenance of every reused V542 helper."""
    return {
        "hierarchical_means": (
            "pes2ts_core.pesn.offline.evaluation.key_atom_protocols."
            "hierarchical_means — frame -> branch -> reaction -> cluster "
            "(with reaction=cluster, branch=0 this is the per-curve frame "
            "mean, exactly the frozen trainer path)"),
        "ordering_pair_accuracy": (
            "pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3."
            "ordering_pair_accuracy"),
        "average_precision / auc_mann_whitney / _sigmoid": (
            "pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 / "
            "key_atom_v52_train_v3"),
        "fixed_coverage_direction_at_3": (
            "NOT imported and NOT a rename of the V542 direction leaf: V542's "
            "direction IS sre_and_sign.sign_accuracy_cluster_median "
            "(scripts/agent_ops/v542_stats.py :177-190, plan §1.5 :108/§2 F3) "
            "while this metric is defined on the teacher's top-3 |Δd| pairs"),
    }


def _aggregation_notes() -> dict[str, str]:
    return {
        "unit": "one row per curve in 'per_curve' (the §9.4 bootstrap cluster)",
        "move.rmse_zero": "median over curves (V542); mean also reported",
        "move.sign_accuracy": "median over curves (V542); mean also reported",
        "move.dead_zone_fraction": "global pair ratio (V542)",
        "move.fixed_coverage.direction_at_3": FIXED_COVERAGE_AGGREGATION,
        "move.fixed_coverage.topk_overlap_at_3": FIXED_COVERAGE_AGGREGATION,
        "move.pair_delta_d.mae": "mean over all changed pairs; curve median too",
        "move.basis_projection_residual": "median over key rows",
        "process.q.mae_curve_median": "median over curves (primary)",
        "process.q.corr_curve_median": "median over eligible curves",
        "process.ordering": "pair-weighted over all (curve, direction) groups",
        "process.endpoint": "frame-pooled over unmasked rows",
        "seed": "one report per (role, seed); §9.2/§9.4 seed aggregation is the "
                "caller's (v543_stats)",
    }


def ambiguities() -> list[str]:
    """Places where the plan's own wording leaves a choice (reported, not hidden)."""
    return [
        "§9.1 :879 names the metric 'fixed_coverage.direction_at_3' while §9.2 "
        ":903 defines k = min(3, n_pairs): this evaluator implements the "
        "formula of §9.2 and reports the effective k of each frame through "
        "the per-frame hit/overlap indices used in the tests.",
        "§9.2 :910 gives the aggregation chain '帧均值 → 曲线均值 → seed 均值' "
        "without saying whether the cross-curve summary is a mean or a median; "
        "this module reports BOTH (direction_at_3 = mean over curves — the "
        "chain's literal reading — and direction_at_3_curve_median). H2/H3 "
        "must pin one before freezing thresholds; both come from the same "
        "per-curve rows so the choice cannot change evidence.",
        "§9.1 :878 'dead_fraction | 预测位移小于阈值的比例' does not define the "
        "threshold or the unit; the frozen V542 H2 leaf is the PAIR dead-zone "
        "fraction (§1.5 :106 sre_and_sign.dead_zone_fraction) while the plan's "
        "wording reads as an ATOM fraction. Both are reported (dead_fraction = "
        f"the V542 leaf; atom_dead_fraction with a {ATOM_DEAD_THRESHOLD} Å "
        "threshold) and the H2 leaf is named explicitly.",
        "§9.1 :891 marks q.corr_curve_median '仅诊断' and §9.4 :1031 forbids the "
        "2-point Pearson, but the V542 leaf computed a correlation on any "
        "curve with >=2 non-degenerate points. corr_curve_median (>=3 points) "
        "is the frozen reading and corr_curve_median_v542_anchor keeps the V542 "
        "number as a historical anchor only.",
        "§9.1 :884 'recall_at_fpr_0p05 | inner-val 上选阈值使 FPR≤0.05' does not "
        "say which threshold wins among the feasible ones; this module "
        "maximises recall and breaks ties towards the HIGHER (more "
        "conservative) threshold, and reports the achieved FPR so a reader can "
        "see it is a constraint, not an equality.",
        "§9.2 :910 asks for a Start/Pre-TS/Post-TS/Endpoint and a "
        "breaking/forming/both breakdown but does not define 'breaking/forming/"
        "both' for FRAMES; the runner assigns a frame class from the signs of "
        "its changed pairs' pair_edit (the §7.1 :640 taxonomy) and the "
        "'unmoved' bucket exists for frames with no changed pair.",
        "the plan does not state the q-validity denominator of frac_curves_ge3 "
        "('curves with q labels' vs 'curves in the frame set'); this module "
        "uses curves that have at least one masked-in q point (the same "
        "denominator as n_eval_curves), so the ratio is self-consistent.",
    ]


__all__ = [
    "ATOM_DEAD_THRESHOLD",
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_MODE_MULTISEED",
    "BOOTSTRAP_MODE_SHARED",
    "BOOTSTRAP_PAIRING",
    "BOOTSTRAP_SEED_AGGREGATION",
    "BOOTSTRAP_SEED_DERIVATION",
    "BOOTSTRAP_SEED_ROOT",
    "BOOTSTRAP_UNIT",
    "BOND_SCALE_NOTE",
    "CI_HIGH_PERCENTILE",
    "CI_LOW_PERCENTILE",
    "COSINE_SOFT_NORM",
    "DENOM_EPS",
    "ECE_BINS",
    "EFFECTIVE_ATOM_THRESHOLD",
    "EvalFrame",
    "FIXED_COVERAGE_AGGREGATION",
    "FIXED_COVERAGE_TOPK",
    "FPR_CONSTRAINT",
    "FrameTerms",
    "INTENT_CLASSES",
    "PLAN_SECTION",
    "Q_CORR_MIN_POINTS",
    "Q_FRAC_GE3_MIN",
    "REPORT_SCHEMA",
    "RESIDUAL_NORM_FLOOR",
    "SIGN_PAIR_THRESHOLD",
    "STRATA_ORDER",
    "V543EvalError",
    "ambiguities",
    "evaluate_baseline",
    "evaluate_v543",
    "evaluator_spec",
    "evaluator_spec_sha256",
    "exact_delta_d",
    "fixed_coverage_at_k",
    "frame_terms",
    "paired_curve_bootstrap",
    "paired_curve_bootstrap_multiseed",
    "recall_at_fpr",
    "soft_cosine",
    "topk_pair_indices",
    "zero_residual_frames",
]
