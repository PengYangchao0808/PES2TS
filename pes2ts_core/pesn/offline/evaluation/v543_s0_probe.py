"""V5.4.3 WP3 — S0-B Latent Sufficiency Probe + S0-C Vector Basis Coverage.

Two diagnostics of the WP1/WP2 frozen front end, implemented VERBATIM from the
campaign plan (``docs/plans/PES2TS_V5.4.3_双Decoder_50K_全面开发与训练方案_v1.0
_20261102.md``):

S0-B — Latent Sufficiency Probe (§7.1, lines 614–630)
    "冻结 Encoder，只训练很小的线性/MLP 诊断头" with the four probes of the
    plan's table,字面照抄:

    ====  =========================  ==============  =========================
    id    input                      target          purpose (plan text)
    ====  =========================  ==============  =========================
    B1    ``graph_s`` only           ``q``/endpoint  "全局标量是否足够 PROCESS"
    B2    ``graph_s+key_s+key_v``    ``δu = y - u0`` "Bridge 是否包含 MOVE 残差信息"
    B3    ``key_s + key_v``          ``δu``          "局部信息是否足够"
    B4    ``graph_s + key_s``        ``δu``          "向量基底是否有额外贡献"
    ====  =========================  ==============  =========================

    Discipline (plan :625–630): probes train on INNER-TRAIN only, a
    curve-level probe-val cut of inner-train selects the epoch, inner-val is
    evaluated exactly once, results are diagnostic only (never champion
    selection), and the report carries residual MAE / R² / per-stratum and
    per-reaction-type breakdowns against the M0 ``‖y-u0‖`` baseline.  The
    plan's Go condition is quoted as a NUMBER here, not interpreted
    (``S0_B_GO_RELATIVE_IMPROVEMENT = 0.10``); the go/no-go decision itself is
    the orchestrator's, never this module's.

S0-C — Vector Basis Coverage (§7.1, lines 632–642)
    For the real teacher residual ``r = y - u0``, per key atom: ``rank(key_v)``,
    min/max singular value, condition number, projection residual
    ``‖r_perp‖/‖r‖`` (with channel 0 AND without channel 0), teacher
    coefficient magnitudes, the leave-one-channel coverage change, stratified
    by Start/Pre-TS/Post-TS/Endpoint and by breaking/forming/both.  The plan's
    interpretation discipline is recorded as data: "8 个任意 3D 向量几乎必然
    张成 R3，所以「覆盖率≈100%」不是好消息也不是坏消息"; the frozen thresholds
    are ``S0_C_CONDITION_THRESHOLD = 100.0`` ("条件数 >100") and
    ``S0_C_PROJECTION_RESIDUAL_THRESHOLD = 0.5`` ("投影残差中位数 >0.5") — this
    module REPORTS the comparison and never rules.

Contract anchors (why the code looks the way it does):

* §5.2 field table :282–284 — ``graph_s [G,64]`` (A0 ``z_head``), ``key_s
  [K,48]`` (last-layer ``h``), ``key_v [K,8,3]`` (Bridge basis).  The probe
  reads its inputs from a :class:`~pes2ts_core.pesn.models.v543.contracts.
  DecoderContextV1` — the frozen WP2 surface — and never from a batch.
* §5.3 :355–393 — the eight ``key_v`` channels; channel 0 is
  ``u0[k]/a_ref`` (hence the "with / without channel 0" pair of §7.1 :637).
* §2 F7 / 附录 E #4 (:1659) — "8 通道基底覆盖率 覆盖率 + 条件数 + 有效秩 + 留一
  + probe"; a lone coverage number is inflating by construction, so every
  coverage figure here is emitted next to condition number, effective rank,
  teacher coefficients, leave-one-out sensitivity and the S0-B probe.
* §7.1 S0 出口 :644–649 — S0-A/B/C all land in the ``encoder_bridge_parity``
  report; any failure reworks WP2.
* ADR-0034 §10 v1.1 (2026-11-02) — ``pair_edit = BOND_SCALE × forward ΔB``
  (``BOND_SCALE = 3.0``).  The S0-B targets here are ``δu`` in Å, **not**
  ``pair_edit``, so no scale factor may be silently learned; the one place
  ``pair_edit`` is read at all is the breaking/forming/both label of S0-C,
  where only its SIGN is used (sign-invariant under the positive
  ``BOND_SCALE``).  Both facts are recorded in the emitted report
  (``bond_scale_note``) instead of being assumed by the reader.

Supervision discipline (hard constraint, plan §5.6 :420–425)
    ``y_teacher`` / ``pair_index`` / ``pair_edit`` / the frame-mask records are
    passed to the functions of this module as EXPLICIT, SEPARATE arguments —
    they are offline analysis inputs.  Nothing here ever puts a supervision
    tensor into a ``DecoderContextV1`` or into an encoder forward; the probe
    heads are trained OUTSIDE the encoder graph (``torch.no_grad`` features).

Everything in this module is pure/offline: no file IO, no dataset access, no
checkpoint loading (that plumbing lives in ``scripts/agent_ops/
v543_s0_probe.py``, mirroring ``v542_a0_parity.py``).  It is therefore
CPU-fast and testable on synthetic tensors, which is what
``tests/pesn/test_v543_s0_probe.py`` does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Final, Mapping, Sequence

import torch
from torch import nn

from pes2ts_core.pesn.features.key_atom_spec_v3 import BOND_SCALE
from pes2ts_core.pesn.models.v543.contracts import (
    GRAPH_S_DIM_V543,
    KEY_S_DIM_V543,
    KEY_V_CHANNELS_V543,
    KEY_V_DIM_V543,
    V543Error,
)

# ---------------------------------------------------------------------------
# typed refusal (code/detail; same taxonomy as the WP1/WP2 modules)
# ---------------------------------------------------------------------------
class V543ProbeError(V543Error):
    """An S0-B/S0-C diagnostic refused to produce a number."""


# ---------------------------------------------------------------------------
# plan constants (every number below is quoted, never invented)
# ---------------------------------------------------------------------------
#: Plan §7.1 :630 — "**Go 条件**：B2 的残差 MAE 明显低于 B1/B4（例如相对 B4
#: 改善 ≥10%）".  Stored as the plan's own example figure; the verdict is the
#: orchestrator's.
S0_B_GO_RELATIVE_IMPROVEMENT: Final[float] = 0.10
#: Plan §7.1 :642 — "若条件数 >100 或 投影残差中位数 >0.5，Bridge 需要重新设计".
S0_C_CONDITION_THRESHOLD: Final[float] = 100.0
S0_C_PROJECTION_RESIDUAL_THRESHOLD: Final[float] = 0.5

#: Plan §7.1 :636 — ``rank`` / min-max singular value / condition number.
#: Numerical rank tolerance: ``tol = rtol · s_max`` with
#: ``rtol = max(m, n) · eps`` — the numpy ``matrix_rank`` convention (which is
#: also ``torch.linalg.matrix_rank``'s default).  Spelled out here so the
#: reported rank is reproducible without reading torch's defaults.
def rank_rtol(shape: Sequence[int]) -> float:
    """``max(m, n) · eps(float32)`` for a ``[..., m, n]`` basis."""
    return float(max(int(shape[-2]), int(shape[-1]))) * float(
        torch.finfo(torch.float32).eps)


RANK_RTOL_FACTOR: Final[str] = "max(m, n) * eps(float32)"

#: Plan §7.1 :640 — the four frozen strata (``v541_strata.STRATA_ORDER_V541``).
STRATA_ORDER: Final[tuple[str, ...]] = ("start", "pre_TS", "post_TS",
                                        "endpoint")
#: Plan §7.1 :640 — "按 breaking/forming/both 分层".  ``unmoved`` is an
#: implementation-only bucket for key atoms with no incident changed pair (it
#: cannot be called breaking/forming/both); it is always reported, never
#: folded into another class.
INTENT_CLASSES: Final[tuple[str, ...]] = ("breaking", "forming", "both",
                                          "unmoved")

#: §5.2 field widths (imported, never retyped).
GRAPH_S_DIM: Final[int] = GRAPH_S_DIM_V543
KEY_S_DIM: Final[int] = KEY_S_DIM_V543
KEY_V_CHANNELS: Final[int] = KEY_V_CHANNELS_V543
KEY_V_DIM: Final[int] = KEY_V_DIM_V543

#: The probe feature names, exactly the plan §7.1 :618–623 columns.
FEATURE_GRAPH_S: Final[str] = "graph_s"
FEATURE_KEY_S: Final[str] = "key_s"
FEATURE_KEY_V: Final[str] = "key_v"
PROBE_FEATURES: Final[tuple[str, ...]] = (
    FEATURE_GRAPH_S, FEATURE_KEY_S, FEATURE_KEY_V)

#: Probe targets.
TARGET_Q: Final[str] = "q"
TARGET_ENDPOINT: Final[str] = "endpoint"
TARGET_DELTA_U: Final[str] = "delta_u"


@dataclass(frozen=True)
class ProbeSpec:
    """One probe of the plan §7.1 table (input set + target + purpose text)."""

    probe_id: str
    features: tuple[str, ...]
    targets: tuple[str, ...]
    row_level: str                     # "frame" (B1) or "key" (B2–B4)
    purpose: str                       # plan §7.1 :618–623, verbatim
    plan_lines: str                    # "§7.1 :620"

    def feature_dim(self) -> int:
        """Concatenated input width (before standardisation)."""
        return sum(_feature_width(name) for name in self.features)


def _feature_width(name: str) -> int:
    if name == FEATURE_GRAPH_S:
        return GRAPH_S_DIM
    if name == FEATURE_KEY_S:
        return KEY_S_DIM
    if name == FEATURE_KEY_V:
        return KEY_V_CHANNELS * KEY_V_DIM
    raise V543ProbeError(
        "V543_PROBE_FEATURE",
        f"unknown probe feature {name!r}; known {list(PROBE_FEATURES)}")


#: The four probes,字面照方案 §7.1 :618–623 (order = plan table order).
PROBE_SPECS: Final[Mapping[str, ProbeSpec]] = {
    "B1": ProbeSpec(
        probe_id="B1",
        features=(FEATURE_GRAPH_S,),
        targets=(TARGET_Q, TARGET_ENDPOINT),
        row_level="frame",
        purpose="全局标量是否足够 PROCESS",
        plan_lines="§7.1 :620",
    ),
    "B2": ProbeSpec(
        probe_id="B2",
        features=(FEATURE_GRAPH_S, FEATURE_KEY_S, FEATURE_KEY_V),
        targets=(TARGET_DELTA_U,),
        row_level="key",
        purpose="Bridge 是否包含 MOVE 残差信息",
        plan_lines="§7.1 :621",
    ),
    "B3": ProbeSpec(
        probe_id="B3",
        features=(FEATURE_KEY_S, FEATURE_KEY_V),
        targets=(TARGET_DELTA_U,),
        row_level="key",
        purpose="局部信息是否足够",
        plan_lines="§7.1 :622",
    ),
    "B4": ProbeSpec(
        probe_id="B4",
        features=(FEATURE_GRAPH_S, FEATURE_KEY_S),
        targets=(TARGET_DELTA_U,),
        row_level="key",
        purpose="向量基底是否有额外贡献",
        plan_lines="§7.1 :623",
    ),
}
#: Plan order of the probe table (report ordering).
PROBE_ORDER: Final[tuple[str, ...]] = ("B1", "B2", "B3", "B4")


# ---------------------------------------------------------------------------
# probe training schedule (the plan fixes no schedule for the tiny heads —
# these are the documented implementation choices of WP3, frozen as constants
# so the report can quote them verbatim)
# ---------------------------------------------------------------------------
#: Hidden width of the two-layer MLP head ("很小的线性/MLP 诊断头").
PROBE_HIDDEN: Final[int] = 64
#: AdamW learning rate for the diagnostic head.
PROBE_LR: Final[float] = 1.0e-3
#: Mini-batch size (rows, not frames).
PROBE_BATCH_SIZE: Final[int] = 512
#: Maximum epochs per fit; the probe-val cut early-stops inside this budget.
PROBE_MAX_EPOCHS: Final[int] = 150
#: Early-stop patience (epochs without probe-val improvement).
PROBE_PATIENCE: Final[int] = 25
#: Weight decay — ZERO on purpose: the probe is a sufficiency diagnostic, a
#: regulariser would shrink exactly the signal we are asking about.
PROBE_WEIGHT_DECAY: Final[float] = 0.0
#: Initialisation seeds; every probe is fitted once per init seed so the
#: reported numbers carry an init-noise band instead of one lucky draw.
PROBE_INIT_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
#: Feature standardisation floor (avoids a divide-by-~zero column).
PROBE_STD_FLOOR: Final[float] = 1.0e-6
#: Report schema (mirrors ``pesn_v543_pair_set_audit_report_v1`` naming).
REPORT_SCHEMA: Final[str] = "pesn_v543_s0_probe_report_v1"
#: Promotion-relevant plan lines the report cites.
PLAN_SECTION: Final[str] = ("§7.1 S0-B :614-630 / S0-C :632-642 / "
                            "S0 出口 :644-649 / §2 F7 + 附录 E #4 :1659")

#: ADR-0034 §10 v1.1 — the frozen scale relation, recorded as DATA.
BOND_SCALE_NOTE: Final[Mapping[str, Any]] = {
    "bond_scale": float(BOND_SCALE),
    "relation": "pair_edit = BOND_SCALE * forward ΔB",
    "s0_b_targets": (
        "δu = y_teacher - u0 in Å — NOT pair_edit, so the probe must not "
        "learn a 3× scale factor and none is applied"),
    "s0_c_pair_edit_use": (
        "sign only (breaking/forming/both); sign is invariant under the "
        "positive BOND_SCALE, no conversion applied"),
}


# ---------------------------------------------------------------------------
# row containers
# ---------------------------------------------------------------------------
@dataclass
class ProbeRows:
    """Row-aligned probe inputs + targets + stratifiers (offline only).

    ``kind`` is ``"frame"`` (one row per frame — B1) or ``"key"`` (one row per
    key atom — B2/B3/B4).  ``features``/``targets`` are keyed by the names of
    :data:`PROBE_SPECS`; the float tensors are ``[N, d]`` on CPU.
    """

    kind: str
    features: dict[str, torch.Tensor] = field(default_factory=dict)
    targets: dict[str, torch.Tensor] = field(default_factory=dict)
    #: 0-based index of the frame this row belongs to (key rows repeat it).
    frame_index: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.int64))
    n_frames: int = 0
    #: per-row stratifiers (plan §7.1 :629 / :640); plain python strings.
    strata: tuple[str, ...] = ()
    intent_classes: tuple[str, ...] = ()
    #: per-row MOVE eligibility (``label_status == "ok"``) — recorded, not
    #: re-applied here (callers filter before appending).
    eligible: torch.Tensor = field(
        default_factory=lambda: torch.zeros(0, dtype=torch.bool))

    def __post_init__(self) -> None:
        if self.kind not in ("frame", "key"):
            raise V543ProbeError(
                "V543_PROBE_ROW_KIND",
                f"row kind must be 'frame' or 'key', got {self.kind!r}")
        n = self.n_rows
        if self.strata and len(self.strata) != n:
            raise V543ProbeError(
                "V543_PROBE_ROW_COUNT",
                f"strata has {len(self.strata)} entries for {n} rows")
        if self.intent_classes and len(self.intent_classes) != n:
            raise V543ProbeError(
                "V543_PROBE_ROW_COUNT",
                f"intent_classes has {len(self.intent_classes)} entries for "
                f"{n} rows")

    @property
    def n_rows(self) -> int:
        if self.targets:
            return int(next(iter(self.targets.values())).shape[0])
        if self.features:
            return int(next(iter(self.features.values())).shape[0])
        return 0

    def check(self) -> "ProbeRows":
        """Validate row alignment; typed refusal on drift."""
        n = self.n_rows
        for name, tensor in self.features.items():
            if tensor.ndim != 2 or int(tensor.shape[0]) != n:
                raise V543ProbeError(
                    "V543_PROBE_SHAPE",
                    f"feature {name!r} must be [N, d] with N={n}, got "
                    f"{list(tensor.shape)}")
            if not bool(torch.isfinite(tensor).all()):
                raise V543ProbeError(
                    "V543_PROBE_NONFINITE", f"feature {name!r} is non-finite")
        for name, tensor in self.targets.items():
            if tensor.ndim != 2 or int(tensor.shape[0]) != n:
                raise V543ProbeError(
                    "V543_PROBE_SHAPE",
                    f"target {name!r} must be [N, c] with N={n}, got "
                    f"{list(tensor.shape)}")
            if not bool(torch.isfinite(tensor).all()):
                raise V543ProbeError(
                    "V543_PROBE_NONFINITE", f"target {name!r} is non-finite")
        if int(self.frame_index.numel()) != n:
            raise V543ProbeError(
                "V543_PROBE_SHAPE",
                f"frame_index must have N={n} entries, got "
                f"{int(self.frame_index.numel())}")
        return self


def concat_probe_rows(parts: Sequence[ProbeRows]) -> ProbeRows:
    """Concatenate same-kind rows (seed pooling); frames are re-indexed."""
    if not parts:
        raise V543ProbeError("V543_PROBE_EMPTY", "no probe rows to pool")
    kinds = {part.kind for part in parts}
    if len(kinds) != 1:
        raise V543ProbeError(
            "V543_PROBE_ROW_KIND",
            f"cannot pool mixed row kinds {sorted(kinds)}")
    kind = kinds.pop()
    names_f = sorted(set().union(*(set(p.features) for p in parts)))
    names_t = sorted(set().union(*(set(p.targets) for p in parts)))
    features: dict[str, torch.Tensor] = {}
    targets: dict[str, torch.Tensor] = {}
    for name in names_f:
        present = [p for p in parts if name in p.features]
        if len(present) != len(parts):
            raise V543ProbeError(
                "V543_PROBE_FEATURE",
                f"feature {name!r} is missing from some pooled parts")
        features[name] = torch.cat([p.features[name] for p in parts], dim=0)
    for name in names_t:
        present = [p for p in parts if name in p.targets]
        if len(present) != len(parts):
            raise V543ProbeError(
                "V543_PROBE_TARGET",
                f"target {name!r} is missing from some pooled parts")
        targets[name] = torch.cat([p.targets[name] for p in parts], dim=0)

    offset = 0
    frame_parts: list[torch.Tensor] = []
    strata: list[str] = []
    intents: list[str] = []
    eligible: list[torch.Tensor] = []
    for part in parts:
        frame_parts.append(part.frame_index + offset)
        offset += int(part.n_frames)
        strata.extend(part.strata)
        intents.extend(part.intent_classes)
        eligible.append(part.eligible)
    return ProbeRows(
        kind=kind, features=features, targets=targets,
        frame_index=torch.cat(frame_parts, dim=0) if frame_parts
        else torch.zeros(0, dtype=torch.int64),
        n_frames=offset, strata=tuple(strata), intent_classes=tuple(intents),
        eligible=torch.cat(eligible, dim=0) if eligible
        else torch.zeros(0, dtype=torch.bool),
    ).check()


def take_probe_rows(rows: ProbeRows, mask: torch.Tensor) -> ProbeRows:
    """Row subset by a boolean mask (stratifier strings follow)."""
    mask = mask.to(torch.bool).reshape(-1)
    if int(mask.numel()) != rows.n_rows:
        raise V543ProbeError(
            "V543_PROBE_ROW_COUNT",
            f"mask has {int(mask.numel())} entries for {rows.n_rows} rows")
    index = torch.nonzero(mask, as_tuple=False).reshape(-1)
    return ProbeRows(
        kind=rows.kind,
        features={k: v[index] for k, v in rows.features.items()},
        targets={k: v[index] for k, v in rows.targets.items()},
        frame_index=rows.frame_index[index],
        n_frames=rows.n_frames,
        strata=tuple(rows.strata[int(i)] for i in index)
        if rows.strata else (),
        intent_classes=tuple(rows.intent_classes[int(i)] for i in index)
        if rows.intent_classes else (),
        eligible=rows.eligible[index] if int(rows.eligible.numel())
        else torch.zeros(0, dtype=torch.bool),
    ).check()


def probe_input_matrix(spec: ProbeSpec, rows: ProbeRows) -> torch.Tensor:
    """Concatenate a probe's input columns in the plan's stated order."""
    missing = [name for name in spec.features if name not in rows.features]
    if missing:
        raise V543ProbeError(
            "V543_PROBE_FEATURE",
            f"probe {spec.probe_id} needs features {missing}; rows carry "
            f"{sorted(rows.features)}")
    tensors = [rows.features[name].to(torch.float32) for name in spec.features]
    matrix = torch.cat(tensors, dim=1) if len(tensors) > 1 else tensors[0]
    if int(matrix.shape[1]) != spec.feature_dim():
        raise V543ProbeError(
            "V543_PROBE_SHAPE",
            f"probe {spec.probe_id} input must be [N,{spec.feature_dim()}], "
            f"got {list(matrix.shape)}")
    return matrix


# ---------------------------------------------------------------------------
# the diagnostic head (tiny MLP; plan §7.1 :616 "很小的线性/MLP 诊断头")
# ---------------------------------------------------------------------------
class ProbeHead(nn.Module):
    """``Linear(d,h) → ReLU → Linear(h,out)`` — the whole probe model."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int = PROBE_HIDDEN,
                 *, dtype: torch.dtype = torch.float32) -> None:
        super().__init__()
        if int(in_dim) <= 0 or int(out_dim) <= 0:
            raise V543ProbeError(
                "V543_PROBE_SHAPE",
                f"probe head needs in/out > 0, got {in_dim}/{out_dim}")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden = int(hidden)
        self.fc1 = nn.Linear(int(in_dim), int(hidden), dtype=dtype)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(int(hidden), int(out_dim), dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))

    def param_count(self) -> int:
        return int(sum(p.numel() for p in self.parameters()))


# ---------------------------------------------------------------------------
# feature standardisation (frozen from the TRAIN rows only)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Standardizer:
    """Column z-score fitted on train rows (never on probe-val / inner-val)."""

    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def fit(cls, matrix: torch.Tensor) -> "Standardizer":
        if matrix.ndim != 2 or int(matrix.shape[0]) == 0:
            raise V543ProbeError(
                "V543_PROBE_EMPTY",
                f"cannot fit a standardizer on {list(matrix.shape)}")
        mean = matrix.mean(dim=0)
        std = matrix.std(dim=0, unbiased=False).clamp_min(PROBE_STD_FLOOR)
        return cls(mean=mean, std=std)

    def apply(self, matrix: torch.Tensor) -> torch.Tensor:
        return (matrix - self.mean) / self.std


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def regression_metrics(pred: torch.Tensor, target: torch.Tensor,
                       *, prefix: str = "") -> dict[str, float]:
    """MAE / RMSE / R² of a probe prediction.

    The plan §7.1 :629 asks for "残差 MAE、R²".  The frozen V5.3-MP MOVE loss
    measures each key atom by its displacement NORM (``key_atom_losses_v3``
    :428 ``e = (u - y).norm(dim=-1)``), so the headline MAE here is the mean
    per-key L2 norm ``mean_k ‖pred - target‖`` in Å — directly comparable to
    the M0 baseline ``mean_k ‖y - u0‖``.  The per-axis MAE, per-axis RMSE and
    the element-wise R² are reported alongside for completeness.
    """
    if pred.shape != target.shape:
        raise V543ProbeError(
            "V543_PROBE_SHAPE",
            f"prediction/target shape mismatch: {list(pred.shape)} vs "
            f"{list(target.shape)}")
    diff = pred - target
    per_row_norm = diff.norm(dim=-1)
    ss_res = float((diff ** 2).sum())
    ss_tot = float(((target - target.mean(dim=0)) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else float("nan")
    return {
        f"{prefix}mae_atom_angstrom": float(per_row_norm.mean()),
        f"{prefix}mae_component_angstrom": float(diff.abs().mean()),
        f"{prefix}rmse_angstrom": math.sqrt(ss_res / max(diff.numel(), 1)),
        f"{prefix}r2": r2,
        f"{prefix}n_rows": float(int(target.shape[0])),
    }


def baseline_metrics(target: torch.Tensor,
                     *, prefix: str = "m0_") -> dict[str, float]:
    """The M0 baseline of plan §7.1 :629 — "与 M0 的 ``‖y-u0‖`` 基线比较".

    M0 predicts ``u = u0``, i.e. ``δu = 0``; the residual MAE is then
    ``mean_k ‖y - u0‖``.
    """
    zero = torch.zeros_like(target)
    return regression_metrics(zero, target, prefix=prefix)


# ---------------------------------------------------------------------------
# probe fitting (deterministic; epoch chosen on the probe-val cut)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ProbeTrainConfig:
    hidden: int = PROBE_HIDDEN
    lr: float = PROBE_LR
    batch_size: int = PROBE_BATCH_SIZE
    max_epochs: int = PROBE_MAX_EPOCHS
    patience: int = PROBE_PATIENCE
    weight_decay: float = PROBE_WEIGHT_DECAY
    init_seeds: tuple[int, ...] = PROBE_INIT_SEEDS


@dataclass
class FittedProbe:
    """One fitted diagnostic head + its selection record (never a checkpoint)."""

    spec: ProbeSpec
    target: str
    config: ProbeTrainConfig
    standardizer: Standardizer
    state: dict[str, dict[str, torch.Tensor]]
    selected_epoch: dict[int, int]
    val_history: dict[int, list[float]]
    init_seeds: tuple[int, ...]
    n_train_rows: int
    n_val_rows: int
    param_count: int
    #: Train-set target statistics.  Standardising the TARGET is the only
    #: reason a probe of a 0.05 Å-scale residual converges in a sane number of
    #: epochs; the reported metrics are always in the target's own units (the
    #: inverse transform :meth:`decode` is applied before every measurement).
    target_mean: torch.Tensor = field(
        default_factory=lambda: torch.zeros(1, dtype=torch.float32))
    target_std: torch.Tensor = field(
        default_factory=lambda: torch.ones(1, dtype=torch.float32))

    def predict(self, rows: ProbeRows, *, ensemble: bool = True
                ) -> torch.Tensor:
        """Predictions; ``ensemble=True`` averages the init seeds."""
        matrix = self.standardizer.apply(
            probe_input_matrix(self.spec, rows))
        preds: list[torch.Tensor] = []
        seeds = self.init_seeds if ensemble else self.init_seeds[:1]
        for seed in seeds:
            head = ProbeHead(int(matrix.shape[1]),
                             int(rows.targets[self.target].shape[1]),
                             self.config.hidden)
            head.load_state_dict(self.state[int(seed)])
            head.eval()
            with torch.no_grad():
                preds.append(self.decode(head(matrix)))
        return torch.stack(preds, dim=0).mean(dim=0)

    def decode(self, raw: torch.Tensor) -> torch.Tensor:
        """Undo the train-set target standardisation (back to Å / raw units)."""
        return raw * self.target_std + self.target_mean

    def encode(self, target: torch.Tensor) -> torch.Tensor:
        """Apply the train-set target standardisation (fitting domain)."""
        return (target - self.target_mean) / self.target_std


def _require_target(spec: ProbeSpec, target: str) -> None:
    if target not in spec.targets:
        raise V543ProbeError(
            "V543_PROBE_TARGET",
            f"probe {spec.probe_id} does not target {target!r}; its targets "
            f"are {list(spec.targets)}")


def _check_learnable(train: ProbeRows, val: ProbeRows, target: str) -> None:
    if train.n_rows == 0 or val.n_rows == 0:
        raise V543ProbeError(
            "V543_PROBE_EMPTY",
            f"probe rows must be non-empty (train={train.n_rows}, "
            f"val={val.n_rows})")
    for rows in (train, val):
        if target not in rows.targets:
            raise V543ProbeError(
                "V543_PROBE_TARGET",
                f"rows carry no target {target!r}: {sorted(rows.targets)}")
    if int(train.targets[target].shape[1]) != int(val.targets[target].shape[1]):
        raise V543ProbeError(
            "V543_PROBE_SHAPE",
            "train/val target widths differ: "
            f"{int(train.targets[target].shape[1])} vs "
            f"{int(val.targets[target].shape[1])}")


def train_probe(
        spec: ProbeSpec,
        train: ProbeRows,
        val: ProbeRows,
        *,
        target: str,
        config: ProbeTrainConfig | None = None,
) -> FittedProbe:
    """Fit one probe head on ``train``, selecting the epoch on ``val``.

    Deterministic given ``config.init_seeds``: the network init and every
    epoch's row permutation come from ``torch.manual_seed``/``torch.Generator``
    seeded by ``(init_seed, epoch)`` — no global RNG state is consumed.
    """
    cfg = config or ProbeTrainConfig()
    _require_target(spec, target)
    _check_learnable(train, val, target)

    x_train = probe_input_matrix(spec, train)
    y_train = train.targets[target].to(torch.float32)
    x_val = probe_input_matrix(spec, val)
    y_val = val.targets[target].to(torch.float32)
    standardizer = Standardizer.fit(x_train)
    x_train = standardizer.apply(x_train)
    x_val = standardizer.apply(x_val)
    target_scaler = Standardizer.fit(y_train)
    y_train_std = target_scaler.apply(y_train)

    out_dim = int(y_train.shape[1])
    batch = max(1, min(int(cfg.batch_size), int(x_train.shape[0])))
    states: dict[int, dict[str, dict[str, torch.Tensor]]] = {}
    chosen: dict[int, int] = {}
    history: dict[int, list[float]] = {}

    for init_seed in cfg.init_seeds:
        torch.manual_seed(int(init_seed))
        head = ProbeHead(int(x_train.shape[1]), out_dim, int(cfg.hidden))
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=float(cfg.lr),
            weight_decay=float(cfg.weight_decay))
        generator = torch.Generator().manual_seed(int(init_seed))
        best_val = float("inf")
        best_epoch = 0
        best_state = {k: v.detach().clone()
                      for k, v in head.state_dict().items()}
        curves: list[float] = []
        stale = 0
        for epoch in range(int(cfg.max_epochs)):
            head.train()
            order = torch.randperm(int(x_train.shape[0]), generator=generator)
            for start in range(0, int(order.numel()), batch):
                index = order[start:start + batch]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.mse_loss(
                    head(x_train[index]), y_train_std[index])
                loss.backward()
                optimizer.step()
            head.eval()
            with torch.no_grad():
                prediction = head(x_val) * target_scaler.std \
                    + target_scaler.mean
                val_mae = float(
                    (prediction - y_val).norm(dim=-1).mean())
            curves.append(val_mae)
            if val_mae < best_val - 1e-12:
                best_val = val_mae
                best_epoch = epoch + 1
                best_state = {k: v.detach().clone()
                              for k, v in head.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= int(cfg.patience):
                    break
        states[int(init_seed)] = {k: v.detach().clone()
                                  for k, v in best_state.items()}
        chosen[int(init_seed)] = int(best_epoch)
        history[int(init_seed)] = [float(v) for v in curves]

    return FittedProbe(
        spec=spec, target=target, config=cfg, standardizer=standardizer,
        state=states, selected_epoch=chosen, val_history=history,
        init_seeds=tuple(int(s) for s in cfg.init_seeds),
        n_train_rows=int(train.n_rows), n_val_rows=int(val.n_rows),
        param_count=ProbeHead(int(x_train.shape[1]), out_dim,
                              int(cfg.hidden)).param_count(),
        target_mean=target_scaler.mean.clone(),
        target_std=target_scaler.std.clone(),
    )


def evaluate_probe(fitted: FittedProbe, rows: ProbeRows, *,
                   per_init_seed: bool = True) -> dict[str, Any]:
    """Final (inner-val) evaluation of one fitted probe — ONCE per split."""
    target = rows.targets[fitted.target].to(torch.float32)
    y = target
    ensemble = fitted.predict(rows)
    out: dict[str, Any] = {
        "probe_id": fitted.spec.probe_id,
        "target": fitted.target,
        "features": list(fitted.spec.features),
        "feature_dim": int(fitted.spec.feature_dim()),
        "param_count": int(fitted.param_count),
        "n_train_rows": int(fitted.n_train_rows),
        "n_val_rows": int(fitted.n_val_rows),
        "selected_epoch": {str(k): int(v)
                           for k, v in fitted.selected_epoch.items()},
        "metrics_ensemble": regression_metrics(ensemble, y),
        "baseline_m0": baseline_metrics(y),
    }
    if per_init_seed:
        seeds: dict[str, Any] = {}
        for seed in fitted.init_seeds:
            # per-init-seed prediction of THAT seed only (no ensembling)
            matrix = fitted.standardizer.apply(
                probe_input_matrix(fitted.spec, rows))
            head = ProbeHead(int(matrix.shape[1]), int(y.shape[1]),
                             fitted.config.hidden)
            head.load_state_dict(fitted.state[seed])
            head.eval()
            with torch.no_grad():
                pred = fitted.decode(head(matrix))
            seeds[str(seed)] = regression_metrics(pred, y)
        out["metrics_per_init_seed"] = seeds
        maes = [row["mae_atom_angstrom"] for row in seeds.values()]
        out["mae_atom_across_init_seeds"] = {
            "mean": float(sum(maes) / len(maes)),
            "std": float(_std(maes)),
            "min": float(min(maes)),
            "max": float(max(maes)),
        }
    baseline = out["baseline_m0"]["m0_mae_atom_angstrom"]
    ensemble_mae = out["metrics_ensemble"]["mae_atom_angstrom"]
    out["relative_improvement_vs_m0"] = (
        float((baseline - ensemble_mae) / baseline) if baseline > 0 else
        float("nan"))
    return out


def _std(values: Sequence[float]) -> float:
    n = len(values)
    if n <= 1:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / n)


def grouped_metrics(pred: torch.Tensor, target: torch.Tensor,
                    labels: Sequence[str], universe: Sequence[str]
                    ) -> dict[str, dict[str, float]]:
    """Per-bucket MAE/R² + M0 baseline for precomputed predictions.

    Shared by :func:`stratify_metrics` and by the WP3 runner, which pools
    per-seed predictions before stratifying (each probe is fitted on its own
    seed's inner-train rows, so pooling cannot reuse one head).
    """
    if int(pred.shape[0]) != int(target.shape[0]):
        raise V543ProbeError(
            "V543_PROBE_ROW_COUNT",
            f"pred has {int(pred.shape[0])} rows, target "
            f"{int(target.shape[0])}")
    if len(labels) != int(target.shape[0]):
        raise V543ProbeError(
            "V543_PROBE_ROW_COUNT",
            f"{len(labels)} labels for {int(target.shape[0])} rows")
    out: dict[str, dict[str, float]] = {}
    for label in universe:
        index = torch.tensor(
            [i for i, value in enumerate(labels) if value == label],
            dtype=torch.int64)
        if int(index.numel()) == 0:
            out[label] = {"n_rows": 0.0}
            continue
        out[label] = regression_metrics(pred[index], target[index])
        out[label].update(baseline_metrics(target[index], prefix="m0_"))
    return out


def stratify_metrics(fitted: FittedProbe, rows: ProbeRows, *,
                     key: str) -> dict[str, dict[str, float]]:
    """Per-stratum / per-reaction-type breakdown (plan §7.1 :629 / :640).

    ``key`` is ``"stratum"`` (the four frozen strata) or ``"intent"``
    (breaking/forming/both/unmoved).  Every bucket that exists is reported,
    including empty ones (``n_rows = 0``) — a silently dropped stratum would
    hide exactly the systematic-direction failure §6.5 is about.
    """
    if key == "stratum":
        labels = rows.strata
        universe = STRATA_ORDER
    elif key == "intent":
        labels = rows.intent_classes
        universe = INTENT_CLASSES
    else:
        raise V543ProbeError(
            "V543_PROBE_STRATIFIER",
            f"unknown stratifier {key!r}; known ['stratum','intent']")
    if not labels:
        raise V543ProbeError(
            "V543_PROBE_STRATIFIER", f"rows carry no {key} labels")
    return grouped_metrics(
        fitted.predict(rows), rows.targets[fitted.target].to(torch.float32),
        labels, universe)


def go_comparison(s0_b: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Report the plan §7.1 :630 comparison — as MEASUREMENTS, no verdict.

    ``relative improvement`` is ``(MAE_B4 - MAE_B2) / MAE_B4`` on the pooled
    inner-val rows (pooled over seeds), the direction the plan states
    ("B2 的残差 MAE 明显低于 B1/B4 … 相对 B4 改善 ≥10%").  The plan also names
    B1 in that sentence although B1's own targets are ``q``/endpoint, not
    ``δu``; the δu-side reference for B1 is therefore the M0 constant
    predictor ``δu = 0``, which is reported next to B2 (see the report's
    ``ambiguities`` block).
    """
    def mae(probe: str) -> float | None:
        entry = s0_b.get(probe)
        if not entry:
            return None
        metrics = entry.get("metrics_ensemble") or {}
        return metrics.get("mae_atom_angstrom")

    mae_b2, mae_b4, mae_b3 = mae("B2"), mae("B4"), mae("B3")
    b1 = s0_b.get("B1") or {}
    m0 = (b1.get("delta_u_m0_baseline") or {}).get("m0_mae_atom_angstrom")
    out: dict[str, Any] = {
        "plan_lines": "§7.1 :630",
        "go_threshold_relative_improvement": S0_B_GO_RELATIVE_IMPROVEMENT,
        "mae_b2_angstrom": mae_b2,
        "mae_b4_angstrom": mae_b4,
        "mae_b3_angstrom": mae_b3,
        "mae_b1_delta_u_baseline_angstrom": m0,
        "thresholds_only_no_verdict": (
            "every field below is a measurement; the go/no-go decision is "
            "the orchestrator's"),
    }
    if mae_b2 is not None and mae_b4:
        out["relative_improvement_b2_vs_b4"] = float(
            (mae_b4 - mae_b2) / mae_b4)
        out["go_threshold_met"] = bool(
            out["relative_improvement_b2_vs_b4"]
            >= S0_B_GO_RELATIVE_IMPROVEMENT)
    else:
        out["relative_improvement_b2_vs_b4"] = None
        out["go_threshold_met"] = None
    if mae_b2 is not None and m0:
        out["relative_improvement_b2_vs_m0"] = float((m0 - mae_b2) / m0)
    return out


# ---------------------------------------------------------------------------
# S0-C: vector basis coverage (plan §7.1 :632–642)
# ---------------------------------------------------------------------------
@dataclass
class BasisCoverage:
    """Per-key-atom basis diagnostics (all tensors ``[K]`` / ``[K,8]``)."""

    key_v: torch.Tensor                 # [K,8,3] as given
    residual: torch.Tensor              # [K,3] = y - u0
    singular_values: torch.Tensor       # [K,3] descending
    rank: torch.Tensor                  # [K] int64 (numerical rank)
    condition_number: torch.Tensor       # [K] float (inf when s_min == 0)
    effective_rank: torch.Tensor        # [K] participation ratio
    stable_rank: torch.Tensor           # [K] ‖A‖_F²/‖A‖₂²
    coverage: torch.Tensor              # [K] ‖proj‖/‖r‖ (all 8 channels)
    projection_residual_ratio: torch.Tensor  # [K] ‖r_perp‖/‖r‖ (all 8)
    coverage_no_ch0: torch.Tensor        # [K] channels 1..7
    projection_residual_ratio_no_ch0: torch.Tensor
    coefficients: torch.Tensor          # [K,8] least-squares teacher coeffs
    leave_one_out_coverage: torch.Tensor  # [K,8] coverage without channel j
    leave_one_out_delta: torch.Tensor    # [K,8] coverage - coverage_{-j}
    residual_norm: torch.Tensor          # [K] ‖r‖
    valid: torch.Tensor                  # [K] bool (‖r‖ above the floor)

    @property
    def n_keys(self) -> int:
        return int(self.key_v.shape[0])


#: A key row whose teacher residual is below this (Å) has no direction to
#: project; the ratio ``‖r_perp‖/‖r‖`` is undefined there, so the row is
#: excluded from the ratios (and counted in ``n_zero_residual_rows``) instead
#: of contributing a 0/0.  Stated as a constant so the report can quote it.
RESIDUAL_NORM_FLOOR_ANGSTROM: Final[float] = 1.0e-9


def _pinv_projection(basis: torch.Tensor, residual: torch.Tensor
                     ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Least-squares projection of ``residual`` onto ``span(basis columns)``.

    ``basis`` is ``[K,3,C]`` (C ≥ 1).  Returns ``(coeffs [K,C],
    projection [K,3], perp [K,3])`` using the pseudo-inverse, which is the
    minimum-norm least-squares solution when the columns are dependent — the
    honest thing to do on the rank-deficient bases this diagnostic expects.
    """
    pinv = torch.linalg.pinv(basis)                 # [K,C,3]
    coeffs = torch.einsum("kcd,kd->kc", pinv, residual)
    projection = torch.einsum("kdc,kc->kd", basis, coeffs)
    perp = residual - projection
    return coeffs, projection, perp


def basis_coverage(key_v: torch.Tensor, residual: torch.Tensor,
                   *, channel0: bool = True) -> BasisCoverage:
    """The full S0-C per-key computation (plan §7.1 :634–640).

    Args:
        key_v: ``[K,8,3]`` Bridge basis (channel 0 = ``u0/a_ref``, §5.3).
        residual: ``[K,3]`` real teacher residual ``r = y - u0`` (Å).
        channel0: kept for symmetry with the report — the "without channel 0"
            arms are ALWAYS computed; this flag is unused (recorded so the
            caller cannot think the choice is global).

    Refusals: wrong shapes/dtypes, non-finite inputs.
    """
    if key_v.ndim != 3 or int(key_v.shape[1]) != KEY_V_CHANNELS \
            or int(key_v.shape[2]) != KEY_V_DIM:
        raise V543ProbeError(
            "V543_PROBE_SHAPE",
            f"key_v must be [K,{KEY_V_CHANNELS},{KEY_V_DIM}], got "
            f"{list(key_v.shape)}")
    if residual.ndim != 2 or int(residual.shape[1]) != KEY_V_DIM:
        raise V543ProbeError(
            "V543_PROBE_SHAPE",
            f"residual must be [K,{KEY_V_DIM}], got {list(residual.shape)}")
    if int(residual.shape[0]) != int(key_v.shape[0]):
        raise V543ProbeError(
            "V543_PROBE_SHAPE",
            f"key_v has {int(key_v.shape[0])} rows, residual "
            f"{int(residual.shape[0])}")
    del channel0
    key_v = key_v.to(torch.float32)
    residual = residual.to(torch.float32)
    if not bool(torch.isfinite(key_v).all()) or \
            not bool(torch.isfinite(residual).all()):
        raise V543ProbeError(
            "V543_PROBE_NONFINITE", "key_v/residual contain non-finite values")
    n_keys = int(key_v.shape[0])

    # --- singular values / rank / conditioning (plan :636) -----------------
    basis = key_v.transpose(1, 2)          # [K,3,8] — columns are channels
    singular_values = torch.linalg.svdvals(basis)        # [K,3] descending
    s_max = singular_values[:, 0]
    s_min = singular_values[:, -1]
    rtol = rank_rtol(tuple(basis.shape))
    rank = torch.linalg.matrix_rank(basis, rtol=rtol)      # [K] int64
    eps = torch.finfo(torch.float32).eps
    tol = s_max * rtol
    with torch.no_grad():
        condition_number = torch.where(
            s_min > tol, s_max / s_min.clamp_min(eps),
            torch.full_like(s_min, float("inf")))
    sq = singular_values ** 2
    frob2 = sq.sum(dim=1)
    effective_rank = (singular_values.sum(dim=1) ** 2) / frob2.clamp_min(eps)
    stable_rank = frob2 / sq[:, 0].clamp_min(eps)

    # --- projection residual with / without channel 0 (plan :637–638) ------
    residual_norm = residual.norm(dim=-1)
    valid = residual_norm > RESIDUAL_NORM_FLOOR_ANGSTROM

    coeffs_all, proj_all, perp_all = _pinv_projection(basis, residual)
    coverage = _ratio(proj_all.norm(dim=-1), residual_norm, valid)
    projection_residual = _ratio(perp_all.norm(dim=-1), residual_norm, valid)

    basis_no0 = basis[:, :, 1:]
    _c0, proj_no0, perp_no0 = _pinv_projection(basis_no0, residual)
    coverage_no0 = _ratio(proj_no0.norm(dim=-1), residual_norm, valid)
    projection_residual_no0 = _ratio(
        perp_no0.norm(dim=-1), residual_norm, valid)

    # --- leave-one-channel-out coverage change (plan :639) -----------------
    leave_cov = torch.zeros((n_keys, KEY_V_CHANNELS), dtype=torch.float32)
    for channel in range(KEY_V_CHANNELS):
        keep = [j for j in range(KEY_V_CHANNELS) if j != channel]
        basis_drop = basis[:, :, keep]
        _c, proj_drop, _p = _pinv_projection(basis_drop, residual)
        leave_cov[:, channel] = _ratio(
            proj_drop.norm(dim=-1), residual_norm, valid)
    leave_delta = coverage.unsqueeze(-1) - leave_cov

    return BasisCoverage(
        key_v=key_v,
        residual=residual,
        singular_values=singular_values,
        rank=rank,
        condition_number=condition_number,
        effective_rank=effective_rank,
        stable_rank=stable_rank,
        coverage=coverage,
        projection_residual_ratio=projection_residual,
        coverage_no_ch0=coverage_no0,
        projection_residual_ratio_no_ch0=projection_residual_no0,
        coefficients=coeffs_all,
        leave_one_out_coverage=leave_cov,
        leave_one_out_delta=leave_delta,
        residual_norm=residual_norm,
        valid=valid,
    )


def _ratio(numerator: torch.Tensor, denominator: torch.Tensor,
           valid: torch.Tensor) -> torch.Tensor:
    """``num/den`` where ``valid`` (NaN elsewhere — never a silent zero)."""
    out = torch.full_like(numerator, float("nan"))
    if bool(valid.any()):
        out[valid] = numerator[valid] / denominator[valid].clamp_min(
            RESIDUAL_NORM_FLOOR_ANGSTROM)
    return out


@dataclass(frozen=True)
class CoverageSubset:
    """Distribution summary of one coverage/ratio tensor over a row subset."""

    n_rows: int
    mean: float
    median: float
    p05: float
    p25: float
    p75: float
    p95: float
    minimum: float
    maximum: float


def _quantiles(values: torch.Tensor) -> tuple[float, ...]:
    if int(values.numel()) == 0:
        nan = float("nan")
        return (nan,) * 7
    q = torch.quantile(
        values, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95],
                             dtype=torch.float32))
    return (float(values.mean()), float(q[2]), float(q[0]), float(q[1]),
            float(q[3]), float(q[4]), float(values.min()),
            float(values.max()))


def _subset_summary(values: torch.Tensor, valid: torch.Tensor
                    ) -> CoverageSubset:
    """Distribution of ``values`` over rows where ``valid`` holds.

    ``values`` may be ``[K]`` or ``[K,C]``; a ``[K]`` ``valid`` mask is
    broadcast over the trailing axis so a per-channel tensor is summarised
    over exactly the same rows.
    """
    if valid.ndim < values.ndim:
        mask = valid.reshape(valid.shape + (1,) * (values.ndim - valid.ndim))
        mask = mask.expand_as(values)
    else:
        mask = valid
    picked = values[mask & torch.isfinite(values)]
    mean, median, p05, p25, p75, p95, low, high = _quantiles(picked)
    return CoverageSubset(
        n_rows=int(picked.numel()), mean=mean, median=median, p05=p05,
        p25=p25, p75=p75, p95=p95, minimum=low, maximum=high)


def _as_dict(summary: CoverageSubset) -> dict[str, Any]:
    return {
        "n_rows": int(summary.n_rows),
        "mean": summary.mean,
        "median": summary.median,
        "p05": summary.p05,
        "p25": summary.p25,
        "p75": summary.p75,
        "p95": summary.p95,
        "min": summary.minimum,
        "max": summary.maximum,
    }


def summarize_basis_coverage(
        coverage: BasisCoverage,
        *,
        strata: Sequence[str] | None = None,
        intent_classes: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Report-ready S0-C block: distributions + thresholds + strata tables.

    The plan's two rework thresholds are emitted as explicit comparisons
    (:data:`S0_C_CONDITION_THRESHOLD`, :data:`S0_C_PROJECTION_RESIDUAL_
    THRESHOLD`) with the measured median; the decision stays with the
    orchestrator.
    """
    valid = coverage.valid
    n_total = coverage.n_keys
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise V543ProbeError(
            "V543_PROBE_EMPTY",
            "no key row has a teacher residual above "
            f"{RESIDUAL_NORM_FLOOR_ANGSTROM} Å — S0-C has nothing to measure")

    rank_values = coverage.rank
    condition = coverage.condition_number
    cond_valid = condition[valid & torch.isfinite(condition)]
    finite_cond = int(cond_valid.numel())
    over_threshold = int((cond_valid > S0_C_CONDITION_THRESHOLD).sum())
    proj_median = float(coverage.projection_residual_ratio[valid].median())
    proj_median_no0 = float(
        coverage.projection_residual_ratio_no_ch0[valid].median())

    per_channel: list[dict[str, Any]] = []
    for channel in range(KEY_V_CHANNELS):
        delta = coverage.leave_one_out_delta[valid, channel]
        per_channel.append({
            "channel": int(channel),
            "coefficient_abs": _as_dict(_subset_summary(
                coverage.coefficients[:, channel].abs(), valid)),
            "coverage_without_channel": _as_dict(_subset_summary(
                coverage.leave_one_out_coverage[:, channel], valid)),
            "coverage_drop": _as_dict(_subset_summary(
                delta, torch.isfinite(delta))),
            "n_rows_drop_gt_0p05": int((delta > 0.05).sum()),
            "n_rows_drop_gt_0p10": int((delta > 0.10).sum()),
            "share_rows_drop_gt_0p05": float(
                (delta > 0.05).to(torch.float32).mean()),
        })

    out: dict[str, Any] = {
        "n_key_rows": int(n_total),
        "n_valid_rows": int(n_valid),
        "n_zero_residual_rows": int(n_total - n_valid),
        "residual_norm_floor_angstrom": RESIDUAL_NORM_FLOOR_ANGSTROM,
        "residual_norm_angstrom": _as_dict(_subset_summary(
            coverage.residual_norm, valid)),
        "rank": {
            "definition": (
                "torch.linalg.matrix_rank on the [3,8] basis with "
                "rtol = max(3,8)*eps(float32), i.e. tol = s_max*rtol "
                "(the numpy matrix_rank convention)"),
            "counts": {str(r): int((rank_values == r).sum())
                       for r in sorted({int(v) for v in rank_values.tolist()})},
            "mean": float(rank_values.to(torch.float32).mean()),
            "share_full_rank_3": float(
                (rank_values == 3).to(torch.float32).mean()),
        },
        "singular_values": {
            "largest": _as_dict(_subset_summary(
                coverage.singular_values[:, 0], valid)),
            "middle": _as_dict(_subset_summary(
                coverage.singular_values[:, 1], valid)),
            "smallest": _as_dict(_subset_summary(
                coverage.singular_values[:, -1], valid)),
        },
        "condition_number": {
            **_as_dict(_subset_summary(condition, valid)),
            "n_finite": finite_cond,
            "n_infinite": int(n_valid - finite_cond),
            "threshold": S0_C_CONDITION_THRESHOLD,
            "n_rows_over_threshold": over_threshold,
            "share_rows_over_threshold": float(
                over_threshold / max(n_valid, 1)),
            "median_over_threshold": bool(
                _subset_summary(condition, valid).median
                > S0_C_CONDITION_THRESHOLD),
        },
        "effective_rank": {
            "definition": "(Σσ)²/Σσ² participation ratio of the 8 channels",
            **{k: v for k, v in _as_dict(_subset_summary(
                coverage.effective_rank, valid)).items()},
        },
        "stable_rank": {
            "definition": "‖A‖_F²/‖A‖₂² (σ² sum / σ_max²)",
            **_as_dict(_subset_summary(coverage.stable_rank, valid)),
        },
        "coverage": {
            "definition": "‖P r‖/‖r‖ onto all 8 channels (plan :637)",
            **_as_dict(_subset_summary(coverage.coverage, valid)),
        },
        "coverage_no_channel0": {
            "definition": "‖P r‖/‖r‖ onto channels 1..7 (plan :637)",
            **_as_dict(_subset_summary(coverage.coverage_no_ch0, valid)),
        },
        "projection_residual_ratio": {
            "definition": "‖r_perp‖/‖r‖ against all 8 channels (plan :637)",
            **_as_dict(_subset_summary(
                coverage.projection_residual_ratio, valid)),
            "threshold": S0_C_PROJECTION_RESIDUAL_THRESHOLD,
            "median_over_threshold": bool(
                proj_median > S0_C_PROJECTION_RESIDUAL_THRESHOLD),
        },
        "projection_residual_ratio_no_channel0": {
            "definition": "‖r_perp‖/‖r‖ against channels 1..7",
            **_as_dict(_subset_summary(
                coverage.projection_residual_ratio_no_ch0, valid)),
            "threshold": S0_C_PROJECTION_RESIDUAL_THRESHOLD,
            "median_over_threshold": bool(
                proj_median_no0 > S0_C_PROJECTION_RESIDUAL_THRESHOLD),
        },
        "teacher_coefficients": {
            "definition": (
                "minimum-norm least-squares coefficients c of r ≈ Σ_j c_j "
                "key_v[:,j] (pseudo-inverse; plan :638 教师系数幅度分布)"),
            "per_channel": per_channel,
            "all_channels_abs": _as_dict(_subset_summary(
                coverage.coefficients.abs(), valid)),
        },
        "leave_one_channel_out": {
            "per_channel": per_channel,
            "note": (
                "coverage_drop = coverage(all 8) - coverage(without channel "
                "j); a channel the basis cannot spare shows a positive drop "
                "(plan :639 留一通道后的覆盖率变化)"),
        },
        "thresholds_no_verdict": {
            "condition_number_threshold": S0_C_CONDITION_THRESHOLD,
            "projection_residual_median_threshold":
                S0_C_PROJECTION_RESIDUAL_THRESHOLD,
            "measured_condition_number_median": _subset_summary(
                condition, valid).median,
            "measured_projection_residual_median": proj_median,
            "measured_projection_residual_median_no_channel0": proj_median_no0,
        },
    }

    if strata:
        out["by_stratum"] = _grouped_coverage(coverage, list(strata),
                                              STRATA_ORDER)
    if intent_classes:
        out["by_intent_class"] = _grouped_coverage(
            coverage, list(intent_classes), INTENT_CLASSES)
    return out


def _grouped_coverage(coverage: BasisCoverage, labels: Sequence[str],
                      universe: Sequence[str]) -> dict[str, Any]:
    if len(labels) != coverage.n_keys:
        raise V543ProbeError(
            "V543_PROBE_ROW_COUNT",
            f"{len(labels)} labels for {coverage.n_keys} key rows")
    out: dict[str, Any] = {}
    for label in universe:
        index = torch.tensor(
            [i for i, value in enumerate(labels) if value == label],
            dtype=torch.int64)
        if int(index.numel()) == 0:
            out[label] = {"n_key_rows": 0}
            continue
        mask = torch.zeros(coverage.n_keys, dtype=torch.bool)
        mask[index] = True
        mask = mask & coverage.valid
        if int(mask.sum()) == 0:
            out[label] = {"n_key_rows": int(index.numel()),
                          "n_valid_rows": 0}
            continue
        cond = coverage.condition_number[mask]
        cond = cond[torch.isfinite(cond)]
        out[label] = {
            "n_key_rows": int(index.numel()),
            "n_valid_rows": int(mask.sum()),
            "rank_mean": float(
                coverage.rank[mask].to(torch.float32).mean()),
            "condition_number_median": float(cond.median())
            if int(cond.numel()) else float("inf"),
            "condition_number_over_threshold": int(
                (cond > S0_C_CONDITION_THRESHOLD).sum()),
            "coverage_median": float(coverage.coverage[mask].median()),
            "coverage_no_channel0_median": float(
                coverage.coverage_no_ch0[mask].median()),
            "projection_residual_median": float(
                coverage.projection_residual_ratio[mask].median()),
            "projection_residual_median_no_channel0": float(
                coverage.projection_residual_ratio_no_ch0[mask].median()),
            "effective_rank_median": float(
                coverage.effective_rank[mask].median()),
            "teacher_coefficient_abs_median": float(
                coverage.coefficients[mask].abs().median()),
        }
    return out


__all__ = [
    "BOND_SCALE_NOTE",
    "BasisCoverage",
    "CoverageSubset",
    "FEATURE_GRAPH_S",
    "FEATURE_KEY_S",
    "FEATURE_KEY_V",
    "FittedProbe",
    "INTENT_CLASSES",
    "PLAN_SECTION",
    "PROBE_BATCH_SIZE",
    "PROBE_FEATURES",
    "PROBE_HIDDEN",
    "PROBE_INIT_SEEDS",
    "PROBE_LR",
    "PROBE_MAX_EPOCHS",
    "PROBE_ORDER",
    "PROBE_PATIENCE",
    "PROBE_SPECS",
    "PROBE_WEIGHT_DECAY",
    "ProbeHead",
    "ProbeRows",
    "ProbeSpec",
    "ProbeTrainConfig",
    "RANK_RTOL_FACTOR",
    "REPORT_SCHEMA",
    "RESIDUAL_NORM_FLOOR_ANGSTROM",
    "S0_B_GO_RELATIVE_IMPROVEMENT",
    "S0_C_CONDITION_THRESHOLD",
    "S0_C_PROJECTION_RESIDUAL_THRESHOLD",
    "STRATA_ORDER",
    "Standardizer",
    "TARGET_DELTA_U",
    "TARGET_ENDPOINT",
    "TARGET_Q",
    "V543ProbeError",
    "baseline_metrics",
    "basis_coverage",
    "concat_probe_rows",
    "evaluate_probe",
    "go_comparison",
    "grouped_metrics",
    "probe_input_matrix",
    "rank_rtol",
    "regression_metrics",
    "stratify_metrics",
    "summarize_basis_coverage",
    "take_probe_rows",
    "train_probe",
]
