"""V5.4.3 WP3 — real-data runner for the S0-B probe and the S0-C basis audit.

WHAT THIS RUNS (plan §7.1, verbatim line references in
``pes2ts_core/pesn/offline/evaluation/v543_s0_probe.py``):

* **S0-B Latent Sufficiency Probe** (plan :614–630) — for each frozen A0 seed
  (``17/29/43`` per the campaign config ``source.seeds``) the encoder is loaded
  strict (``CoarseEncoder.load``, 45-key payload, ``strict=False`` does not
  exist), the Bridge builds the frozen ``DecoderContextV1``, and four tiny
  diagnostic heads are fitted OUTSIDE the encoder graph:

      B1  graph_s            -> q, endpoint logit      (PROCESS sufficiency)
      B2  graph_s+key_s+key_v -> δu = y - u0           (Bridge has MOVE info)
      B3  key_s+key_v        -> δu                     (local info enough)
      B4  graph_s+key_s      -> δu                     (basis contributes more)

  Probe rows come from the INNER-TRAIN slice only (a deterministic
  curve-level 10 % cut of inner-train is the probe-val epoch selector); the
  inner-val slice is evaluated exactly ONCE, at the end.

* **S0-C Vector Basis Coverage** (plan :632–642) — per key atom of the
  inner-val frames: ``rank(key_v)``, min/max singular value, condition number,
  projection residual ``‖r_perp‖/‖r‖`` with AND without channel 0, teacher
  coefficient magnitudes, leave-one-channel coverage change, stratified by
  Start/Pre-TS/Post-TS/Endpoint and by breaking/forming/both.

The plan fixes no sample budget for either diagnostic, so the budget is an
explicit, reported parameter of this runner (``--train-frames`` /
``--probe-val-frames`` / ``--eval-frames``); every position stays inside the
frozen ``v53mp_pool_split(seed)`` slices and the three row sets are proven
curve-disjoint (a shared curve is a typed ``V543_PROBE_CURVE_LEAK`` refusal).

Supervision discipline (plan §5.6): ``y_teacher``/``pair_index``/``pair_edit``
and the frame ``label_status`` records are read HERE and passed to the probe
library as separate, explicit arguments.  The encoder and the Bridge only ever
receive ``forward_view_v3(batch)`` (the encoder types-refuses a supervision key
anyway); no supervision tensor reaches a context field or a model forward.

CLI::

    PYTHONPATH=. python scripts/agent_ops/v543_s0_probe.py \
        --out outputs/pesn/v543_dual_decoder_50k/audit/s0_probe_report.json

Typed refusals: :class:`~pes2ts_core.pesn.offline.evaluation.v543_s0_probe.
V543ProbeError` for the diagnostic itself, ``V542ParityError`` (imported, not
modified) for the frozen asset/data view, plus yaml/HF availability.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import numpy as np
import torch

from pes2ts_core.pesn.contracts.key_atom_v3 import forward_view_v3
from pes2ts_core.pesn.models.v543 import CoarseEncoder, LatentBridge
from pes2ts_core.pesn.offline.evaluation.v543_s0_probe import (
    BOND_SCALE_NOTE,
    INTENT_CLASSES,
    PLAN_SECTION,
    PROBE_INIT_SEEDS,
    PROBE_MAX_EPOCHS,
    PROBE_ORDER,
    PROBE_SPECS,
    REPORT_SCHEMA,
    RESIDUAL_NORM_FLOOR_ANGSTROM,
    S0_B_GO_RELATIVE_IMPROVEMENT,
    S0_C_CONDITION_THRESHOLD,
    S0_C_PROJECTION_RESIDUAL_THRESHOLD,
    STRATA_ORDER,
    TARGET_DELTA_U,
    TARGET_ENDPOINT,
    TARGET_Q,
    ProbeRows,
    ProbeTrainConfig,
    V543ProbeError,
    basis_coverage,
    concat_probe_rows,
    evaluate_probe,
    go_comparison,
    grouped_metrics,
    regression_metrics,
    summarize_basis_coverage,
    take_probe_rows,
    train_probe,
)
from scripts.agent_ops.v542_a0_parity import (
    DEFAULT_CONFIG as DATA_CONFIG,
    PROJECT_ROOT,
    V542ParityError,
    _open_frozen_dataset,
    load_pinned_assets,
    sha256_file,
)

#: Campaign config: seeds + pinned checkpoint paths/SHAs + data paths.
CAMPAIGN_CONFIG: Final[str] = (
    "config/campaigns/pes2ts_v543_dual_decoder_50k.yaml")
#: Frozen data view (same config the V5.4.2 A0 parity leg pins).
DATA_VIEW_CONFIG: Final[str] = DATA_CONFIG
#: Default report location (plan §11.1 audit directory; schema-styled name).
DEFAULT_OUT: Final[str] = (
    "outputs/pesn/v543_dual_decoder_50k/audit/s0_probe_report.json")
#: Frames per collated batch (the S1 batch width is 8 curves × 4 frames = 32).
FRAMES_PER_BATCH: Final[int] = 32
#: Default sample budget.  The plan §7.1 fixes none; these are the WP3
#: defaults, frozen in the report so a reader can see what was measured.
DEFAULT_TRAIN_FRAMES: Final[int] = 8192
DEFAULT_PROBE_VAL_FRAMES: Final[int] = 2048
DEFAULT_EVAL_FRAMES: Final[int] = 8192
#: Deterministic RNG salts (documented so the sample is reproducible).
TRAIN_SPLIT_SALT: Final[int] = 0x543B01
CURVE_VAL_SALT: Final[str] = "v543|s0probe|probe-val"
#: Share of inner-train CURVES held out for the epoch selector (curve-level,
#: plan :627 "从 inner-train 再切一个 curve-level probe-val 选 epoch").
PROBE_VAL_CURVE_MOD: Final[int] = 10


# ---------------------------------------------------------------------------
# config / asset plumbing (typed refusals, no silent defaults)
# ---------------------------------------------------------------------------
def load_campaign_config(path: Path) -> dict[str, Any]:
    """Read the V543 campaign config ('source' block only, read-only)."""
    import yaml

    if not path.is_file():
        raise V543ProbeError(
            "V543_PROBE_CONFIG", f"campaign config not found: {path}")
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001 - typed conversion
        raise V543ProbeError(
            "V543_PROBE_CONFIG", f"campaign config unreadable: {exc}") from exc
    if not isinstance(cfg, Mapping):
        raise V543ProbeError(
            "V543_PROBE_CONFIG", "campaign config is not a mapping")
    source = cfg.get("source") or {}
    seeds = source.get("seeds")
    checkpoints = source.get("checkpoints") or {}
    if not isinstance(seeds, list) or not seeds:
        raise V543ProbeError(
            "V543_PROBE_CONFIG", f"source.seeds must be a non-empty list, "
            f"got {seeds!r}")
    for seed in seeds:
        entry = checkpoints.get(str(int(seed)))
        if not isinstance(entry, Mapping) or "path" not in entry \
                or "sha256" not in entry:
            raise V543ProbeError(
                "V543_PROBE_CONFIG",
                f"source.checkpoints[{seed}] must pin path+sha256, got "
                f"{entry!r}")
    return dict(cfg)


def resolve_checkpoints(cfg: Mapping[str, Any], seeds: Sequence[int],
                        *, root: Path = PROJECT_ROOT) -> dict[int, dict]:
    """Resolve + SHA-verify the pinned A0 checkpoints for ``seeds``."""
    checkpoints = ((cfg.get("source") or {}).get("checkpoints") or {})
    resolved: dict[int, dict] = {}
    for seed in seeds:
        entry = checkpoints[str(int(seed))]
        rel = str(entry["path"])
        path = Path(rel) if Path(rel).is_absolute() else root / rel
        if not path.is_file():
            raise V543ProbeError(
                "V543_PROBE_CKPT",
                f"A0 checkpoint for seed {seed} not found at {path}")
        actual = sha256_file(path)
        pinned = str(entry["sha256"])
        if actual != pinned:
            raise V543ProbeError(
                "V543_PROBE_CKPT",
                f"A0 checkpoint sha256 mismatch for seed {seed}: {actual} != "
                f"pinned {pinned} ({path})")
        resolved[int(seed)] = {"path": str(path), "sha256": actual}
    return resolved


# ---------------------------------------------------------------------------
# sample selection (deterministic, curve-level disjointness)
# ---------------------------------------------------------------------------
def _curve_bucket(curve_id: str) -> str:
    """Deterministic curve-level bucket: ``probe-val`` or ``train``."""
    import hashlib

    digest = hashlib.sha256(
        f"{CURVE_VAL_SALT}|{curve_id}".encode("utf-8")).hexdigest()
    return "val" if int(digest[:8], 16) % PROBE_VAL_CURVE_MOD == 0 else "train"


@dataclass
class SeedSamples:
    """The three position sets of one seed (all inside the frozen split)."""

    train: list[int]
    probe_val: list[int]
    inner_val: list[int]
    n_train_candidates: int
    selection_budget: tuple[int, int]
    inner_val_coverage: Mapping[str, Any] = field(default_factory=dict)


def draw_probe_samples(dataset: Any, train_pos: Sequence[int],
                       val_pos: Sequence[int], *, seed: int,
                       n_train: int, n_val: int,
                       eval_frames: int) -> SeedSamples:
    """Inner-train (train/probe-val, curve-disjoint) + inner-val positions.

    Inner-val positions are a deterministic UNIFORM SPREAD (``linspace``) over
    the frozen inner-val position list of ``v53mp_pool_split(seed)`` — the plan
    fixes no sampling rule for the S0 diagnostics, and a spread covers the
    whole frozen slice instead of its first curves only.  The slice itself is
    curve-disjoint from inner-train by construction.
    """
    train_pos = np.sort(np.asarray(train_pos, dtype=np.int64))
    val_pos = np.sort(np.asarray(val_pos, dtype=np.int64))
    if train_pos.size < int(n_train) + int(n_val):
        raise V543ProbeError(
            "V543_PROBE_BUDGET",
            f"inner-train slice for seed {seed} has {train_pos.size} "
            f"positions, need {int(n_train) + int(n_val)}")
    rng = np.random.RandomState(int(seed) ^ TRAIN_SPLIT_SALT)
    want = int(n_train) + int(n_val)
    train_frames: list[int] = []
    val_frames: list[int] = []
    seen_train_curves: set[str] = set()
    seen_val_curves: set[str] = set()
    draw_size = min(int(train_pos.size), int(want))
    attempts = 0
    while (len(train_frames) < int(n_train)
           or len(val_frames) < int(n_val)) and attempts < 8:
        attempts += 1
        candidates = rng.choice(int(train_pos.size), size=draw_size,
                                replace=False)
        for index in candidates:
            position = int(train_pos[int(index)])
            curve_id = str(dataset.samples[position].curve_id)
            if _curve_bucket(curve_id) == "val":
                if len(val_frames) >= int(n_val):
                    continue
                seen_val_curves.add(curve_id)
                val_frames.append(position)
            else:
                if len(train_frames) >= int(n_train):
                    continue
                seen_train_curves.add(curve_id)
                train_frames.append(position)
            if len(train_frames) >= int(n_train) and \
                    len(val_frames) >= int(n_val):
                break
        draw_size = min(int(train_pos.size), draw_size * 2)
    if len(train_frames) < int(n_train) or len(val_frames) < int(n_val):
        raise V543ProbeError(
            "V543_PROBE_BUDGET",
            f"seed {seed}: only {len(train_frames)}/{n_train} train and "
            f"{len(val_frames)}/{n_val} probe-val frames could be drawn from "
            f"{draw_size} candidates; raise the candidate draw or lower the "
            "budget")
    overlap = seen_train_curves & seen_val_curves
    if overlap:                                    # pragma: no cover - guard
        raise V543ProbeError(
            "V543_PROBE_CURVE_LEAK",
            f"probe-val curves also appear in probe-train: {sorted(overlap)}")

    if val_pos.size < int(eval_frames):
        raise V543ProbeError(
            "V543_PROBE_BUDGET",
            f"seed {seed}: the frozen inner-val slice has {val_pos.size} "
            f"positions, need {int(eval_frames)}")
    picks = np.linspace(0, int(val_pos.size) - 1, int(eval_frames))
    inner_val = [int(val_pos[int(round(p))]) for p in picks]

    eval_curves = {str(dataset.samples[p].curve_id) for p in inner_val}
    if eval_curves & (seen_train_curves | seen_val_curves):
        raise V543ProbeError(
            "V543_PROBE_CURVE_LEAK",
            "inner-val shares curves with inner-train: "
            f"{sorted(eval_curves & (seen_train_curves | seen_val_curves))}")

    return SeedSamples(
        train=train_frames, probe_val=val_frames, inner_val=inner_val,
        n_train_candidates=int(train_pos.size),
        selection_budget=(int(eval_frames), 1),
        inner_val_coverage={
            "rule": ("deterministic uniform linspace spread over the sorted "
                     "frozen inner-val positions of v53mp_pool_split(seed)"),
            "n_positions_available": int(val_pos.size),
            "n_positions_used": len(inner_val),
        },
    )


# ---------------------------------------------------------------------------
# frame features (encoder + Bridge, no_grad) and supervision row building
# ---------------------------------------------------------------------------
def label_status_of(records: Sequence[Any]) -> list[str]:
    """``label_status`` per frame (audit-only; the MOVE eligibility rule)."""
    return [str(getattr(record, "label_status", "")) for record in records]


def positions_meta(dataset: Any, positions: Sequence[int]) -> list[dict]:
    """``(curve_id, node_index, direction, stratum)`` per position.

    Strata use the frozen V5.4.1 rule (``v541_strata.classify_stratum_v541``
    via ``strata_of_position_v541``), i.e. the same Start/Pre-TS/Post-TS/
    Endpoint definition plan §7.1 :640 asks for — no second definition here.
    """
    from pes2ts_core.pesn.offline.training.v541_strata import (
        StrataIndex,
        strata_of_position_v541,
    )

    samples = dataset.samples
    index = StrataIndex.build(samples, [int(p) for p in positions],
                              dataset=dataset)
    out: list[dict] = []
    for position in positions:
        curve_id, direction, stratum = strata_of_position_v541(
            samples, index, int(position))
        out.append({"curve_id": curve_id, "direction": direction,
                    "stratum": stratum})
    return out


def intent_class_per_key(batch: Mapping[str, Any], key_index: torch.Tensor
                         ) -> list[str]:
    """breaking/forming/both/unmoved per KEY ROW (plan §7.1 :640).

    Uses the supervision ``pair_edit`` SIGN only (``pair_index`` gives the two
    key-atom rows of each changed pair).  ADR-0034 §10 v1.1: ``pair_edit`` is
    the RAW bond-order edit while the forward view carries ``ΔB =
    pair_edit / BOND_SCALE``; the sign is identical under the positive
    ``BOND_SCALE = 3.0``, so no conversion is applied (and none is needed).
    """
    node_of_slot = [int(v) for v in key_index.tolist()]
    slot_of_node = {node: slot for slot, node in enumerate(node_of_slot)}
    bits = np.zeros((len(node_of_slot),), dtype=np.int8)
    pair_index = batch["pair_index"].long()
    pair_edit = batch["pair_edit"].to(torch.float32)
    for column in range(int(pair_index.shape[1])):
        edit = float(pair_edit[column])
        if edit == 0.0:
            continue
        bit = 1 if edit < 0.0 else 2
        for row in (0, 1):
            slot = slot_of_node.get(int(pair_index[row, column]))
            if slot is not None:
                bits[slot] |= bit
    labels = []
    for value in bits.tolist():
        if value == 1:
            labels.append("breaking")
        elif value == 2:
            labels.append("forming")
        elif value == 3:
            labels.append("both")
        else:
            labels.append("unmoved")
    return labels


@dataclass
class FrameFeatures:
    """Encoder/Bridge output + supervision rows of one position chunk."""

    graph_s: torch.Tensor                  # [G,64]
    key_s: torch.Tensor                    # [K,48]
    key_v: torch.Tensor                    # [K,8,3]
    u0: torch.Tensor                       # [K,3]
    delta_u: torch.Tensor                  # [K,3] = y - u0
    key_frame: torch.Tensor                # [K] frame index within the chunk
    q: torch.Tensor                        # [G]
    endpoint_logit: torch.Tensor           # [G]
    label_status: list[str]
    intent_classes: list[str]              # per KEY ROW
    n_frames: int


def encode_chunk(encoder: CoarseEncoder, bridge: LatentBridge,
                 batch: Mapping[str, Any], records: Sequence[Any]
                 ) -> FrameFeatures:
    """One chunk: ``forward_view_v3`` → packet → context → supervision rows."""
    view = forward_view_v3(batch)               # supervision stripped HERE
    with torch.no_grad():
        packet = encoder(view)
        context = bridge.build_context(
            packet, view,
            encoder_ckpt_sha256=(encoder.load_audit or {}).get(
                "encoder_ckpt_sha256"))
    key_ptr = context.key_ptr.long()
    counts = (key_ptr[1:] - key_ptr[:-1]).clamp(min=0)
    key_frame = torch.repeat_interleave(
        torch.arange(int(counts.numel()), dtype=torch.int64), counts)
    teacher = batch["y_teacher"].to(torch.float32)
    if tuple(teacher.shape) != tuple(context.u0.shape):
        raise V543ProbeError(
            "V543_PROBE_SHAPE",
            f"y_teacher {list(teacher.shape)} and u0 "
            f"{list(context.u0.shape)} are not row aligned")
    return FrameFeatures(
        graph_s=context.graph_s,
        key_s=context.key_s,
        key_v=context.key_v,
        u0=context.u0,
        delta_u=teacher - context.u0,
        key_frame=key_frame,
        q=torch.tanh(packet.q0_raw),
        endpoint_logit=packet.e0,
        label_status=label_status_of(records),
        intent_classes=intent_class_per_key(batch, batch["key_index"]),
        n_frames=int(counts.numel()),
    )


def key_rows(chunks: Sequence[FrameFeatures],
             meta: Sequence[Sequence[dict]]) -> ProbeRows:
    """Assemble B2–B4 rows (one per eligible key atom)."""
    features: dict[str, list[torch.Tensor]] = {
        "graph_s": [], "key_s": [], "key_v": []}
    targets: dict[str, list[torch.Tensor]] = {TARGET_DELTA_U: []}
    frames: list[torch.Tensor] = []
    strata: list[str] = []
    intents: list[str] = []
    eligible: list[torch.Tensor] = []
    offset = 0
    for chunk, chunk_meta in zip(chunks, meta):
        if len(chunk_meta) != chunk.n_frames:
            raise V543ProbeError(
                "V543_PROBE_ROW_COUNT",
                f"{len(chunk_meta)} position records for "
                f"{chunk.n_frames} frames")
        frame_ok = torch.tensor(
            [status == "ok" for status in chunk.label_status],
            dtype=torch.bool)
        broadcast = chunk.graph_s[chunk.key_frame]
        features["graph_s"].append(broadcast)
        features["key_s"].append(chunk.key_s)
        features["key_v"].append(
            chunk.key_v.reshape(int(chunk.key_v.shape[0]), -1))
        targets[TARGET_DELTA_U].append(chunk.delta_u)
        frames.append(chunk.key_frame + offset)
        row_ok = frame_ok[chunk.key_frame]
        eligible.append(row_ok)
        for row in range(int(chunk.key_frame.shape[0])):
            frame = int(chunk.key_frame[row])
            strata.append(str(chunk_meta[frame]["stratum"]))
            intents.append(chunk.intent_classes[row])
        offset += int(chunk.n_frames)
    rows = ProbeRows(
        kind="key",
        features={name: torch.cat(parts, dim=0)
                  for name, parts in features.items()},
        targets={name: torch.cat(parts, dim=0)
                 for name, parts in targets.items()},
        frame_index=torch.cat(frames, dim=0),
        n_frames=offset,
        strata=tuple(strata),
        intent_classes=tuple(intents),
        eligible=torch.cat(eligible, dim=0),
    ).check()
    return take_probe_rows(rows, rows.eligible)


def frame_rows(chunks: Sequence[FrameFeatures],
               meta: Sequence[Sequence[dict]]) -> ProbeRows:
    """Assemble B1 rows (one per frame; PROCESS targets are q / endpoint)."""
    graph_s: list[torch.Tensor] = []
    q: list[torch.Tensor] = []
    endpoint: list[torch.Tensor] = []
    frames: list[torch.Tensor] = []
    strata: list[str] = []
    intents: list[str] = []
    offset = 0
    for chunk, chunk_meta in zip(chunks, meta):
        if len(chunk_meta) != chunk.n_frames:
            raise V543ProbeError(
                "V543_PROBE_ROW_COUNT",
                f"{len(chunk_meta)} position records for "
                f"{chunk.n_frames} frames")
        graph_s.append(chunk.graph_s)
        q.append(chunk.q.reshape(-1, 1))
        endpoint.append(chunk.endpoint_logit.reshape(-1, 1))
        frames.append(torch.arange(chunk.n_frames, dtype=torch.int64) + offset)
        # per-frame intent class = the frame's key-row classes collapsed;
        # B1 is a PROCESS probe, so the frame-level label is the frame's own
        # dominant changed-pair role (breaking/forming/both/unmoved).
        per_frame: dict[int, set[str]] = {}
        for row, label in enumerate(chunk.intent_classes):
            frame = int(chunk.key_frame[row])
            per_frame.setdefault(frame, set()).add(label)
        for frame in range(chunk.n_frames):
            labels = per_frame.get(frame, {"unmoved"})
            strata.append(str(chunk_meta[frame]["stratum"]))
            intents.append(_collapse_intent(labels))
        offset += int(chunk.n_frames)
    return ProbeRows(
        kind="frame",
        features={"graph_s": torch.cat(graph_s, dim=0)},
        targets={TARGET_Q: torch.cat(q, dim=0),
                 TARGET_ENDPOINT: torch.cat(endpoint, dim=0)},
        frame_index=torch.cat(frames, dim=0),
        n_frames=offset,
        strata=tuple(strata),
        intent_classes=tuple(intents),
        eligible=torch.ones(offset, dtype=torch.bool),
    ).check()


def _collapse_intent(labels: Iterable[str]) -> str:
    """Frame-level collapse of its key rows' classes (breaking/forming/both)."""
    unique = {label for label in labels if label != "unmoved"}
    if not unique:
        return "unmoved"
    if len(unique) > 1:
        return "both"
    return next(iter(unique))


def frame_status_crosstab(chunks: Sequence[FrameFeatures],
                          meta: Sequence[Sequence[dict]]) -> dict[str, Any]:
    """``label_status`` counts overall and crossed with the stratum.

    Evidence, not decoration: the MOVE eligibility rule is
    ``label_status == "ok"`` (frozen ``_frame_mask_for``), so if every
    Endpoint frame is ``terminal`` the S0-B/S0-C "Endpoint" stratum is
    structurally empty — a reader must be able to SEE that instead of
    wondering why the bucket reports zero rows.
    """
    totals: dict[str, int] = {}
    crosstab: dict[str, dict[str, int]] = {}
    n_frames = 0
    for chunk, chunk_meta in zip(chunks, meta):
        for frame in range(chunk.n_frames):
            status = chunk.label_status[frame]
            stratum = str(chunk_meta[frame]["stratum"])
            totals[status] = int(totals.get(status, 0)) + 1
            bucket = crosstab.setdefault(stratum, {})
            bucket[status] = int(bucket.get(status, 0)) + 1
            n_frames += 1
    return {"n_frames": int(n_frames), "by_label_status": totals,
            "by_stratum_and_status": crosstab}


# ---------------------------------------------------------------------------
# per-seed run
# ---------------------------------------------------------------------------
@dataclass
class SeedResult:
    seed: int
    rows_train_key: ProbeRows
    rows_val_key: ProbeRows
    rows_eval_key: ProbeRows
    rows_train_frame: ProbeRows
    rows_val_frame: ProbeRows
    rows_eval_frame: ProbeRows
    coverage: Any
    eval_key_v: torch.Tensor
    eval_residual: torch.Tensor
    eval_strata: tuple[str, ...]
    eval_intents: tuple[str, ...]
    n_positions: Mapping[str, int]
    inner_val_coverage: Mapping[str, Any]
    #: Frame-level ``label_status`` evidence per row set (train/probe-val/eval).
    status_crosstab: Mapping[str, Any]


def _build_chunks(encoder: CoarseEncoder, bridge: LatentBridge, dataset: Any,
                  lru: Any, positions: Sequence[int]
                  ) -> tuple[list[FrameFeatures], list[list[dict]]]:
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        _batch_from_positions,
    )

    chunks: list[FrameFeatures] = []
    metas: list[list[dict]] = []
    meta = positions_meta(dataset, positions)
    for start in range(0, len(positions), FRAMES_PER_BATCH):
        chunk_positions = list(positions[start:start + FRAMES_PER_BATCH])
        batch, records = _batch_from_positions(
            dataset, lru, chunk_positions, "cpu")
        chunks.append(encode_chunk(encoder, bridge, batch, records))
        metas.append(meta[start:start + FRAMES_PER_BATCH])
    return chunks, metas


def run_seed(*, seed: int, checkpoint: Mapping[str, Any], dataset: Any,
             train_pos: Sequence[int], val_pos: Sequence[int],
             n_train: int, n_val: int, n_eval: int) -> SeedResult:
    """Load the A0 encoder of ``seed`` and run both diagnostics' row build."""
    from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
        _FeatureLRU,
    )

    samples = draw_probe_samples(
        dataset, train_pos, val_pos, seed=int(seed), n_train=int(n_train),
        n_val=int(n_val), eval_frames=int(n_eval))
    all_positions = samples.train + samples.probe_val + samples.inner_val
    lru = _FeatureLRU(dataset, capacity=len(all_positions) + 128)

    encoder = CoarseEncoder.load(str(checkpoint["path"]), int(seed))
    encoder.eval()
    bridge = LatentBridge(
        encoder_ckpt_sha256=str(checkpoint["sha256"]))

    train_chunks, train_meta = _build_chunks(
        encoder, bridge, dataset, lru, samples.train)
    val_chunks, val_meta = _build_chunks(
        encoder, bridge, dataset, lru, samples.probe_val)
    eval_chunks, eval_meta = _build_chunks(
        encoder, bridge, dataset, lru, samples.inner_val)

    rows_train_key = key_rows(train_chunks, train_meta)
    rows_val_key = key_rows(val_chunks, val_meta)
    rows_eval_key = key_rows(eval_chunks, eval_meta)
    rows_train_frame = frame_rows(train_chunks, train_meta)
    rows_val_frame = frame_rows(val_chunks, val_meta)
    rows_eval_frame = frame_rows(eval_chunks, eval_meta)

    coverage = basis_coverage(
        rows_eval_key.features["key_v"].reshape(-1, 8, 3),
        rows_eval_key.targets[TARGET_DELTA_U])

    return SeedResult(
        seed=int(seed),
        rows_train_key=rows_train_key, rows_val_key=rows_val_key,
        rows_eval_key=rows_eval_key,
        rows_train_frame=rows_train_frame, rows_val_frame=rows_val_frame,
        rows_eval_frame=rows_eval_frame,
        coverage=coverage,
        eval_key_v=rows_eval_key.features["key_v"].reshape(-1, 8, 3),
        eval_residual=rows_eval_key.targets[TARGET_DELTA_U],
        eval_strata=rows_eval_key.strata,
        eval_intents=rows_eval_key.intent_classes,
        n_positions={"train": len(samples.train),
                     "probe_val": len(samples.probe_val),
                     "inner_val": len(samples.inner_val),
                     "inner_train_candidates":
                         samples.n_train_candidates},
        inner_val_coverage=samples.inner_val_coverage,
        status_crosstab={
            "train": frame_status_crosstab(train_chunks, train_meta),
            "probe_val": frame_status_crosstab(val_chunks, val_meta),
            "inner_val": frame_status_crosstab(eval_chunks, eval_meta),
            "n_key_rows_all": {
                "train": int(rows_train_key.n_rows
                             + _excluded_rows(train_chunks)),
                "probe_val": int(rows_val_key.n_rows
                                 + _excluded_rows(val_chunks)),
                "inner_val": int(rows_eval_key.n_rows
                                 + _excluded_rows(eval_chunks)),
            },
            "n_key_rows_eligible": {
                "train": int(rows_train_key.n_rows),
                "probe_val": int(rows_val_key.n_rows),
                "inner_val": int(rows_eval_key.n_rows),
            },
        },
    )


def _excluded_rows(chunks: Sequence[FrameFeatures]) -> int:
    """Key rows dropped by the MOVE eligibility mask (label_status != "ok")."""
    dropped = 0
    for chunk in chunks:
        frame_ok = torch.tensor(
            [status == "ok" for status in chunk.label_status],
            dtype=torch.bool)
        dropped += int((~frame_ok[chunk.key_frame]).sum())
    return dropped


# ---------------------------------------------------------------------------
# report assembly
# ---------------------------------------------------------------------------
def _probe_block(results: Sequence[SeedResult], probe_id: str,
                 target: str, config: ProbeTrainConfig
                 ) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor,
                            ProbeRows]:
    """Fit one probe per seed, then pool the inner-val predictions."""
    spec = PROBE_SPECS[probe_id]
    kind = "frame" if spec.row_level == "frame" else "key"
    per_seed: list[dict[str, Any]] = []
    pooled_pred: list[torch.Tensor] = []
    pooled_target: list[torch.Tensor] = []
    pooled_rows: list[ProbeRows] = []
    for result in results:
        if kind == "frame":
            train, val, test = (result.rows_train_frame,
                                result.rows_val_frame,
                                result.rows_eval_frame)
        else:
            train, val, test = (result.rows_train_key, result.rows_val_key,
                                result.rows_eval_key)
        fitted = train_probe(spec, train, val, target=target, config=config)
        metrics = evaluate_probe(fitted, test)
        metrics["seed"] = int(result.seed)
        per_seed.append(metrics)
        pooled_pred.append(fitted.predict(test))
        pooled_target.append(test.targets[target].to(torch.float32))
        pooled_rows.append(test)

    pred = torch.cat(pooled_pred, dim=0)
    tgt = torch.cat(pooled_target, dim=0)
    rows = concat_probe_rows(pooled_rows)
    pooled = regression_metrics(pred, tgt)
    from pes2ts_core.pesn.offline.evaluation.v543_s0_probe import (
        baseline_metrics,
    )

    pooled_baseline = baseline_metrics(tgt)
    baseline_mae = pooled_baseline["m0_mae_atom_angstrom"]
    pooled_mae = pooled["mae_atom_angstrom"]
    block = {
        "probe_id": probe_id,
        "plan_lines": spec.plan_lines,
        "purpose": spec.purpose,
        "features": list(spec.features),
        "feature_dim": int(spec.feature_dim()),
        "target": target,
        "row_level": spec.row_level,
        "param_count": int(per_seed[0]["param_count"]),
        "metrics_pooled_inner_val": pooled,
        "baseline_m0_pooled": pooled_baseline,
        "relative_improvement_vs_m0_pooled": (
            float((baseline_mae - pooled_mae) / baseline_mae)
            if baseline_mae > 0 else float("nan")),
        "metrics_by_seed": per_seed,
        "by_stratum_pooled": grouped_metrics(
            pred, tgt, rows.strata, STRATA_ORDER) if rows.strata else {},
        "by_intent_class_pooled": grouped_metrics(
            pred, tgt, rows.intent_classes,
            INTENT_CLASSES) if rows.intent_classes else {},
        "n_rows_pooled": int(tgt.shape[0]),
    }
    return block, pred, tgt, rows


def build_report(*, seeds: Sequence[int], checkpoints: Mapping[int, dict],
                 results: Sequence[SeedResult], budget: Mapping[str, int],
                 config: ProbeTrainConfig) -> dict[str, Any]:
    """The full ``pesn_v543_s0_probe_report_v1`` report (no verdicts)."""
    s0_b: dict[str, Any] = {}
    # --- B2/B3/B4: δu probes ------------------------------------------------
    for probe_id in ("B2", "B3", "B4"):
        block, _pred, _tgt, _rows = _probe_block(
            results, probe_id, TARGET_DELTA_U, config)
        s0_b[probe_id] = block
    # --- B1: PROCESS scalars (q, endpoint logit) ----------------------------
    spec = PROBE_SPECS["B1"]
    b1: dict[str, Any] = {
        "probe_id": "B1",
        "plan_lines": spec.plan_lines,
        "purpose": spec.purpose,
        "features": list(spec.features),
        "feature_dim": int(spec.feature_dim()),
        "row_level": spec.row_level,
        "targets": [
            {name: _probe_block(results, probe_id, name, config)[0]}
            for probe_id, name in (("B1", TARGET_Q), ("B1", TARGET_ENDPOINT))
        ],
        "note": (
            "B1's stated targets are q and endpoint (§7.1 :620), not δu; it "
            "therefore has no δu MAE of its own. The δu reference for the "
            "plan's 'B2 < B1' clause is the M0 constant predictor δu = 0, "
            "reported as 'delta_u_m0_baseline' below and inside every B2–B4 "
            "'baseline_m0_pooled' block."),
    }
    # B1 rows are per FRAME; the δu baseline is per KEY row — pooled over the
    # same inner-val frames, so the comparison is on identical frames.
    delta_rows = concat_probe_rows([r.rows_eval_key for r in results])
    from pes2ts_core.pesn.offline.evaluation.v543_s0_probe import (
        baseline_metrics,
    )

    b1["delta_u_m0_baseline"] = baseline_metrics(
        delta_rows.targets[TARGET_DELTA_U])
    b1["n_rows_pooled"] = int(delta_rows.n_rows)
    s0_b["B1"] = b1

    # --- the plan's Go-condition comparison (measurements only) ------------
    go_input: dict[str, Any] = {
        probe: {"metrics_ensemble": block.get("metrics_pooled_inner_val")}
        for probe, block in s0_b.items() if probe != "B1"
    }
    go_input["B1"] = {"delta_u_m0_baseline": b1["delta_u_m0_baseline"]}
    go = go_comparison(go_input)
    # pooled MAE fields the comparison quotes, keyed by probe
    go["mae_b2_angstrom"] = s0_b["B2"]["metrics_pooled_inner_val"][
        "mae_atom_angstrom"]
    go["mae_b3_angstrom"] = s0_b["B3"]["metrics_pooled_inner_val"][
        "mae_atom_angstrom"]
    go["mae_b4_angstrom"] = s0_b["B4"]["metrics_pooled_inner_val"][
        "mae_atom_angstrom"]

    # --- S0-C: basis coverage, pooled + per seed --------------------------
    coverage_pooled = basis_coverage(
        torch.cat([r.eval_key_v for r in results], dim=0),
        torch.cat([r.eval_residual for r in results], dim=0))
    strata_pooled = tuple(s for r in results for s in r.eval_strata)
    intents_pooled = tuple(s for r in results for s in r.eval_intents)
    pooled_summary = summarize_basis_coverage(
        coverage_pooled, strata=strata_pooled,
        intent_classes=intents_pooled)
    leave_rows = pooled_summary["leave_one_channel_out"]["per_channel"]
    s0_c: dict[str, Any] = {
        "plan_lines": "§7.1 :632-642",
        "n_seeds": len(results),
        "pooled_inner_val": pooled_summary,
        "by_seed": {
            str(r.seed): summarize_basis_coverage(
                r.coverage, strata=r.eval_strata,
                intent_classes=r.eval_intents)
            for r in results
        },
        "f7_bundle": {
            "note": (
                "§2 F7 / 附录 E #4 :1659 — a lone 'coverage' number is "
                "inflating by construction; it is only readable next to "
                "condition number, effective rank, teacher coefficients, "
                "leave-one-out sensitivity and the S0-B probe, all of which "
                "are in this report"),
            "coverage_median": pooled_summary["coverage"]["median"],
            "coverage_no_channel0_median":
                pooled_summary["coverage_no_channel0"]["median"],
            "condition_number_median":
                pooled_summary["condition_number"]["median"],
            "rank_mean": pooled_summary["rank"]["mean"],
            "effective_rank_median": pooled_summary["effective_rank"]["median"],
            "stable_rank_median": pooled_summary["stable_rank"]["median"],
            "teacher_coefficient_abs_median":
                pooled_summary["teacher_coefficients"]["all_channels_abs"][
                    "median"],
            "leave_one_out_max_median_drop": max(
                row["coverage_drop"]["median"] for row in leave_rows),
            "leave_one_out_most_sensitive_channel": max(
                leave_rows, key=lambda row: row["coverage_drop"]["median"])[
                    "channel"],
        },
    }

    report: dict[str, Any] = {
        "schema": REPORT_SCHEMA,
        "plan_section": PLAN_SECTION,
        "bridge_spec_sha256": LatentBridge().bridge_spec_sha256(),
        "seeds": [int(s) for s in seeds],
        "checkpoints": {str(seed): dict(checkpoints[int(seed)])
                        for seed in seeds},
        "budget": dict(budget),
        "probe_training": {
            "hidden": int(config.hidden),
            "lr": float(config.lr),
            "batch_size": int(config.batch_size),
            "max_epochs": int(config.max_epochs),
            "patience": int(config.patience),
            "weight_decay": float(config.weight_decay),
            "init_seeds": [int(s) for s in config.init_seeds],
            "note": (
                "the plan fixes no schedule for the tiny heads (§7.1 :616 "
                "'很小的线性/MLP 诊断头'); these are the WP3 implementation "
                "choices, identical for B1–B4 so the probe comparison is "
                "capacity-fair"),
        },
        "probe_specs": {
            probe: {
                "features": list(PROBE_SPECS[probe].features),
                "targets": list(PROBE_SPECS[probe].targets),
                "row_level": PROBE_SPECS[probe].row_level,
                "purpose": PROBE_SPECS[probe].purpose,
                "plan_lines": PROBE_SPECS[probe].plan_lines,
            }
            for probe in PROBE_ORDER
        },
        "s0_b": s0_b,
        "s0_b_go_comparison": go,
        "s0_c": s0_c,
        "eligibility": {
            "rule": (
                "MOVE eligibility (frozen _frame_mask_for): a frame's key rows "
                "enter B2–B4 and S0-C only when label_status == 'ok'; B1 uses "
                "every frame because its targets are the PROCESS scalars"),
            "by_seed": {str(r.seed): dict(r.status_crosstab)
                        for r in results},
            "note": (
                "the stratum x label_status cross-tab is evidence: e.g. an "
                "'endpoint' bucket with n_key_rows = 0 is structurally empty "
                "because Endpoint frames are 'terminal' (no y_teacher), not a "
                "sampling accident"),
        },
        "thresholds": {
            "b2_vs_b4_relative_improvement_go": S0_B_GO_RELATIVE_IMPROVEMENT,
            "condition_number_rework": S0_C_CONDITION_THRESHOLD,
            "projection_residual_median_rework":
                S0_C_PROJECTION_RESIDUAL_THRESHOLD,
            "no_verdict": (
                "every threshold above is quoted from the plan; the go/no-go "
                "decision is the orchestrator's, not this report's"),
        },
        "bond_scale_note": dict(BOND_SCALE_NOTE),
        "residual_norm_floor_angstrom": RESIDUAL_NORM_FLOOR_ANGSTROM,
        "frozen_metric_definitions": {
            "mae_atom_angstrom": (
                "mean over key rows of ‖prediction - target‖₂ in Å — the "
                "MOVE-loss geometry (key_atom_losses_v3 :428 "
                "'e = (u - y).norm(dim=-1)')"),
            "baseline_m0": (
                "M0 predicts u = u0 → δu = 0, so its MAE is mean ‖y - u0‖ "
                "(plan §7.1 :629)"),
            "r2": "1 - SS_res/SS_tot over all elements (pooled)",
        },
        "implementation_choices": _implementation_choices(),
        "ambiguities": _ambiguities(),
    }
    return report


def _implementation_choices() -> list[str]:
    """Every place WP3 had to choose because the plan is silent."""
    return [
        "sample budget: the plan fixes none for S0-B/S0-C; the runner's "
        f"--train-frames/--probe-val-frames/--eval-frames defaults "
        f"({DEFAULT_TRAIN_FRAMES}/{DEFAULT_PROBE_VAL_FRAMES}/"
        f"{DEFAULT_EVAL_FRAMES}) are reported in 'budget'.",
        "probe rows come from the frozen inner-train slice of "
        "v53mp_pool_split(seed), drawn by numpy RandomState(seed ^ "
        f"{TRAIN_SPLIT_SALT}) over the inner-train POSITION list (uniform over "
        "frames, not the §7.2 stratified sampler, which exists for training "
        "and is untouched here).",
        "the curve-level probe-val cut is a deterministic SHA-256 bucket "
        f"(1/{PROBE_VAL_CURVE_MOD} of inner-train curves), not RNG; the "
        "runner enforces zero curve overlap between probe-train, probe-val "
        "and inner-val and refuses otherwise.",
        "probe head: Linear(d,64)-ReLU-Linear(64,out), AdamW lr 1e-3, "
        "weight decay 0, batch 512 rows, ≤150 epochs, patience 25, epoch "
        "chosen on the probe-val cut. Three init seeds (0/1/2) per encoder "
        "seed; the report carries the ensemble value AND the per-init-seed "
        "band.",
        "B1 targets q = tanh(q0_raw) and the endpoint logit e0 (never "
        "sigmoid'ed, plan §4.3), both taken from the frozen packet — the "
        "probe only re-reads them from graph_s.",
        "B1 rows are ALL inner-val frames (PROCESS endpoint statuses include "
        "masked/terminal per the campaign config), while B2–B4 rows use the "
        "MOVE eligibility rule label_status == 'ok' (plan §4.1 / the frozen "
        "_frame_mask_for). The excluded-class counts are reported.",
        "residual MAE = mean per-key L2 norm in Å (MOVE-loss geometry); "
        "per-axis MAE, RMSE and pooled R² are reported alongside.",
        "S0-C is computed on the same inner-val key rows that S0-B evaluates "
        "(teacher residual is only defined where a teacher exists).",
        "breaking/forming/both is assigned PER KEY ROW from the signs of its "
        "incident changed pairs (pair_index + pair_edit sign). A key row with "
        "no incident changed pair is reported as 'unmoved' — a fourth bucket "
        "the plan does not name, kept separate rather than folded in.",
        "numerical rank uses rtol = s_max * max(3,8) * eps (numpy "
        "matrix_rank's rule); condition number is s_max/s_min, reported as "
        "inf when the smallest singular value falls under that tolerance.",
        "projection/coefficients use the minimum-norm least-squares solution "
        "(pseudo-inverse) so rank-deficient bases are handled honestly.",
    ]


def _ambiguities() -> list[str]:
    """Places where the plan's own wording is not self-consistent."""
    return [
        "§7.1 :630 names B1 in the Go comparison ('B2 的残差 MAE 明显低于 "
        "B1/B4'), but §7.1 :620 gives B1 the targets q/endpoint — B1 has no "
        "δu MAE. WP3 therefore implements B1 literally (q/endpoint) and "
        "reports the M0 constant predictor δu = 0 as the δu-side reference; "
        "the B2-vs-B4 improvement the plan quantifies ('相对 B4 改善 ≥10%') is "
        "reported exactly as written.",
        "§7.1 :629 asks for a 'by stratum / reaction type' breakdown but does "
        "not define '反应类型'; WP3 uses the breaking/forming/both classes of "
        "§7.1 :640 (the only reaction-type taxonomy in the plan) plus the "
        "'unmoved' bucket for key rows with no incident changed pair.",
        "§7.1 :636 asks for 'rank' without a tolerance convention; the "
        "numerical-rank rule is stated explicitly in the report.",
        "§7.1 :637 '含通道 0 / 去掉通道 0 两版' — both versions are always "
        "computed; there is no mode switch.",
        "§7.1 fixes no seed set for S0-B/S0-C. WP3 uses the campaign's three "
        "frozen encoder seeds (17/29/43, config source.seeds) so that every "
        "probe runs on the exact encoder it will be used with.",
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_seeds(text: str) -> list[int]:
    seeds = [int(token) for token in str(text).replace(",", " ").split()]
    if not seeds:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return seeds


def run(args: argparse.Namespace) -> dict[str, Any]:
    """The whole WP3 measurement (no verdict, no champion selection)."""
    campaign = load_campaign_config(PROJECT_ROOT / CAMPAIGN_CONFIG)
    seeds = [int(s) for s in (args.seeds or campaign["source"]["seeds"])]
    checkpoints = resolve_checkpoints(campaign, seeds)
    data_cfg, _assets = load_pinned_assets(PROJECT_ROOT / DATA_VIEW_CONFIG)
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

    config = ProbeTrainConfig(
        init_seeds=tuple(int(s) for s in (args.init_seeds
                                          or PROBE_INIT_SEEDS)),
        max_epochs=int(args.max_epochs or PROBE_MAX_EPOCHS))
    budget = {
        "train_frames": int(args.train_frames),
        "probe_val_frames": int(args.probe_val_frames),
        "eval_frames": int(args.eval_frames),
        "frames_per_batch": FRAMES_PER_BATCH,
        "probe_val_curve_mod": PROBE_VAL_CURVE_MOD,
        "inner_val_rule": (
            "deterministic uniform linspace spread over the sorted frozen "
            "inner-val positions of v53mp_pool_split(seed), truncated to "
            "--eval-frames positions"),
    }

    results: list[SeedResult] = []
    for seed in seeds:
        train_pos, val_pos = v53mp_pool_split(
            dataset.samples, pool_curve_ids=pool_ids, seed=int(seed))
        result = run_seed(
            seed=int(seed), checkpoint=checkpoints[int(seed)],
            dataset=dataset, train_pos=train_pos,
            val_pos=val_pos, n_train=int(args.train_frames),
            n_val=int(args.probe_val_frames), n_eval=int(args.eval_frames))
        results.append(result)
        print(
            f"[s0probe] seed {seed}: train {result.n_positions['train']} "
            f"frames -> {result.rows_train_key.n_rows} key rows, "
            f"probe-val {result.n_positions['probe_val']} frames -> "
            f"{result.rows_val_key.n_rows} key rows, inner-val "
            f"{result.n_positions['inner_val']} frames -> "
            f"{result.rows_eval_key.n_rows} key rows",
            file=sys.stderr, flush=True)

    report = build_report(
        seeds=seeds, checkpoints=checkpoints, results=results, budget=budget,
        config=config)
    report["ok"] = True
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="V5.4.3 WP3 — S0-B Latent Sufficiency Probe + S0-C Basis "
                    "Coverage (measurements only; no go/no-go verdict)")
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"report path (default {DEFAULT_OUT})")
    parser.add_argument("--seeds", type=_parse_seeds, default=None,
                        help="encoder seeds (default: the campaign config's)")
    parser.add_argument("--train-frames", type=int,
                        default=DEFAULT_TRAIN_FRAMES)
    parser.add_argument("--probe-val-frames", type=int,
                        default=DEFAULT_PROBE_VAL_FRAMES)
    parser.add_argument("--eval-frames", type=int, default=DEFAULT_EVAL_FRAMES)
    parser.add_argument("--init-seeds", type=_parse_seeds, default=None,
                        help="probe init seeds (default 0 1 2)")
    parser.add_argument("--max-epochs", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        report = run(args)
    except (V543ProbeError, V542ParityError) as error:
        print(f"[s0probe] REFUSED {error}", file=sys.stderr)
        return 2
    target = Path(args.out)
    if not target.is_absolute():
        target = PROJECT_ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True,
                                 ensure_ascii=False), encoding="utf-8")
    print(f"[s0probe] report -> {target}", file=sys.stderr)
    return 0


if __name__ == "__main__":                       # pragma: no cover
    raise SystemExit(main())
