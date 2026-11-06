"""V5.4.3 S1 trainer — single-track, decoder-only, frozen-encoder training
(plan §7.2 :651–733, §7.3 :735–746, §7.4 :748–773, §8.1 :797–809, §11.2
``v543_train.py``; WP4).

Normative sources, quoted as implemented
----------------------------------------

Fixed schedule (§7.2 :655–672), all values ``Final`` below and asserted on
construction:

======================  ============================================
item                    value (plan / campaign YAML)
======================  ============================================
updates                 **37,900 flat** (:659)
warmup                  400 updates, linear (:660)
LR                      3e-4 constant after warmup (:661)
optimizer               AdamW (:662)
weight decay            1e-4 (:663)
grad clip               1.0 (:664)
precision               fp32 (:665)
batch                   8 curves × 4 frames = 32 (:666)
sampler                 ``v541_curve_stratified_v1``, RNG ``(seed, update, salt)`` (:667)
validation              every 100 updates, inner-val cap 8,192 (:668)
checkpoint              the track's inner-val primary loss only (:669)
seeds                   17 / 29 / 43 (:670)
resume                  only from a matching ``checkpoint_last.pt`` (:671)
forbidden               ``torch.compile``, dose/frame/layer reduction (:672)
======================  ============================================

Losses (§7.2 :688–711), every coefficient frozen:

* ``L_M = L_atom + 0.5 * L_pair + 0.02 * L_cos`` (:691) — computed by calling
  the FROZEN :func:`~pes2ts_core.pesn.models.key_atom_loss_v53mp_v3.
  key_atom_loss_v53mp` with ``lambda_rank = lambda_q = lambda_endpoint = 0``,
  ``lambda_strong = lambda_dead = 0`` (the V542 recipe, :695) and the frozen
  ``KeyAtomLossV3Config()`` (``lambda_pair = 0.5``) + ``lambda_cos = 0.02``.
  The formula is NOT re-implemented here (:698 "不重新实现不同版本的损失").
* ``L_P = 0.5 * L_ranking + 0.5 * L_q + 0.012385049119773344 * L_endpoint``
  (:703) — computed by calling the FROZEN ``ranking_loss_v53mp`` /
  ``progress_loss_v52`` / ``endpoint_loss_v53mp`` directly, NEVER the MOVE loss
  (:706).
* **PROCESS frame mask** (:707): "PROCESS 的 frame mask 不能沿用
  ``_frame_mask_for``：必须使用 endpoint 标签表的有效 mask，覆盖
  ``ok/terminal/masked/conflict``；terminal 行只要 endpoint 标签有效就进入
  训练".  :func:`process_frame_mask_v543` IS that mask (an
  ``endpoint_label ∈ {0,1}`` row mask from the SAME frozen
  ``_endpoint_for_batch`` helper) and it is applied to the ranking pairs, the q
  rows and the endpoint rows; ``_frame_mask_for`` is used by the MOVE track
  ONLY.  A terminal frame with a valid endpoint label therefore trains the
  PROCESS head while being excluded from MOVE — the F5 correction, asserted by
  ``tests/pesn/test_v543_train.py::test_process_mask_covers_terminal_rows``.
* Skip rule (:711): a batch with no valid supervision for the current track
  records ``skipped_no_grad`` and is not resampled.
* Gradient isolation (:713–726): :meth:`V543Trainer._backward` runs the plan's
  assertion block after EVERY backward, on both tracks.
* Logs / checkpoints (:728–733): ``run_records.jsonl`` (every ``val_every``),
  ``train_log.jsonl`` (every update), ``checkpoint_best.pt`` (inner-val primary
  loss), ``checkpoint_last.pt`` (every ``checkpoint_every`` + at the end),
  ``metrics.json``.
* Resume identity (:750–772): :data:`RESUME_IDENTITY_GROUPS_V543` is the plan's
  14-line list verbatim; :func:`identity_block_v543` writes the flattened keys
  (``decoder_id`` / ``track`` / ``model_version`` are three keys inside the
  plan's third line) and :func:`assert_resume_identity_v543` compares them
  key-by-key PLUS the canonical digest — a mismatch is ``invalid_run``
  (``V543_RESUME_IDENTITY``), never a silent fresh start.

Design decisions and their basis
--------------------------------

1. **The frozen loss functions are called, never copied.**  Every coefficient
   reachable from the plan is a ``Final`` here and is CHECKED against the V5.4.2
   module constants at construction (:func:`assert_frozen_loss_coefficients`),
   so a drifted V542 module is a typed refusal instead of a silent re-tuning.
2. **Supervision is read on a separate code path from the context.**  The
   encoder/Bridge/decoder see only ``forward_view_v3(batch)``; the labels
   (``labels_v52`` / ``endpoint_labels`` / ``pair_index`` / ``pair_edit`` /
   ``y_teacher`` / ``frame_weight``) are read by :func:`_supervision_for_batch`
   and never touch the context — the structural separation of plan §7.2 :684.
3. **A zero-parameter baseline is a real training run.**  M0/P0 have no
   parameter, so no optimizer exists and no backward runs; the loop still
   forwards, computes the frozen loss, validates and checkpoints, producing the
   T0 baseline numbers through the SAME code path a candidate uses
   (§8.1 :799 "M0/P0 三 seed parity、基线锚定").  ``grad_finite`` is recorded as
   ``null`` for such a run so a reader cannot mistake it for a measured
   gradient.
4. **Cache on, autograd off the cache.**  ``frozen_encoder`` mode may read the
   §7.3 cache; a full hit means ZERO encoder forwards for that update, a miss
   means exactly ONE forward for the whole batch (plan §4.4 boundary 1 — never
   a per-frame loop).  ``joint_finetune`` forbids the cache outright
   (:class:`~...v543_feature_cache.V543CacheError`).
5. **Determinism is by construction, not by seeding.**  Every per-update draw
   comes from ``(seed, update, salt)`` (frozen sampler RNG), so a resumed run
   reproduces the uninterrupted one update-for-update (plan §7.4 :773); the
   torch RNG state is snapshotted anyway for anything a WP5/WP6 candidate may
   add.

Discipline: no V5.4.2/V5.3-MP file is modified (plan §11.3 禁止); the frozen
functions are IMPORTED.  No ``torch.compile`` (asserted).  Python 3.12."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping, Sequence

import torch

from pes2ts_core.pesn.contracts.key_atom_v3 import (
    NODE_DEGK_KEY,
    NODE_KEY_SET_KEY,
    NODE_POS_KEY,
    forward_view_v3,
)
from pes2ts_core.pesn.models.key_atom_loss_v53mp_v3 import (
    RANK_MARGIN_V53MP,
    endpoint_loss_v53mp,
    key_atom_loss_v53mp,
    ranking_loss_v53mp,
)
from pes2ts_core.pesn.models.key_atom_loss_v52_v3 import progress_loss_v52
from pes2ts_core.pesn.models.key_atom_losses_v3 import KeyAtomLossV3Config
from pes2ts_core.pesn.models.v543.assembly import V543Assembly
from pes2ts_core.pesn.models.v543.contracts import (
    DECODER_CONTEXT_SCHEMA_V543,
    PRECISION_VERSION_V543,
    V543Error,
)
from pes2ts_core.pesn.models.v543.latent_bridge import BRIDGE_SPEC_SHA256
from pes2ts_core.pesn.models.v543.model import (
    V543Model,
    check_external_output,
)
from pes2ts_core.pesn.models.v543.registry import TRACKS_V543, TRACK_MOVE, TRACK_PROCESS
from pes2ts_core.pesn.offline.data.v543_feature_cache import (
    CACHE_MODE_FROZEN_ENCODER,
    CacheIdentity,
    FrameRecord,
    PacketCache,
    concat_contexts,
    frame_cache_key,
    sha256_text,
    split_context_by_graph,
)
from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
    _FeatureLRU,
    _batch_from_positions,
    _endpoint_for_batch,
    _frame_mask_for,
    _key_ptr,
    _labels_for_batch,
    _update_rng,
    sample_order_pairs,
    v53mp_pool_split,
)
from pes2ts_core.pesn.offline.training.v541_identity import (
    env_state,
    rng_state_restore,
    rng_state_snapshot,
)
from pes2ts_core.pesn.offline.training.v541_strata import (
    SAMPLER_ID_V541,
    StrataIndex,
    batch_rng_v541,
    curve_batch_positions_v541,
    sampler_spec_sha256,
)
from pes2ts_core.pesn.offline.training.v542_train import (
    V542_LAMBDA_ENDPOINT,
    V542_LAMBDA_Q,
    V542_LAMBDA_RANK,
    V542_SAMPLER_SPEC_SHA256_PINNED,
)
from pes2ts_core.pesn.models.v542.identity import canonical_sha256

# ---------------------------------------------------------------------------
# frozen constants (plan §7.2 :655–672, campaign YAML ``training:``)
# ---------------------------------------------------------------------------
#: Checkpoint / identity schema id of this trainer.
V543_TRAIN_SCHEMA: Final[str] = "pesn_v543_train_v1"

#: Flat dose and its warmup window (:659/:660).
V543_TOTAL_UPDATES: Final[int] = 37_900
V543_WARMUP_UPDATES: Final[int] = 400

#: Optimiser constants (:661–664).
V543_LR: Final[float] = 3.0e-4
V543_WEIGHT_DECAY: Final[float] = 1.0e-4
V543_GRAD_CLIP: Final[float] = 1.0

#: Batch geometry (:666).
V543_BATCH_CURVES: Final[int] = 8
V543_FRAMES_PER_CURVE: Final[int] = 4
V543_BATCH_FRAMES: Final[int] = V543_BATCH_CURVES * V543_FRAMES_PER_CURVE

#: Validation / evaluation constants (:668).
V543_VAL_EVERY: Final[int] = 100
V543_INNER_VAL_MAX_SAMPLES: Final[int] = 8_192

#: Checkpoint cadence (:732 "``checkpoint_last.pt``：每 500 updates + 结束").
V543_CHECKPOINT_EVERY: Final[int] = 500

#: Frozen seeds (:670).
V543_FROZEN_SEEDS: Final[tuple[int, ...]] = (17, 29, 43)

#: Sampler id + spec digest (:667 / campaign YAML ``training.sampler*``).
V543_SAMPLER_ID: Final[str] = SAMPLER_ID_V541
V543_SAMPLER_SPEC_SHA256_PINNED: Final[str] = V542_SAMPLER_SPEC_SHA256_PINNED

#: MOVE loss coefficients (:691 "``L_M = L_atom + 0.5 * L_pair + 0.02 * L_cos``").
V543_LAMBDA_PAIR: Final[float] = 0.5
V543_LAMBDA_COS: Final[float] = 0.02
#: Aux arms are OFF in the V542 recipe (:695 "``lambda_strong = lambda_dead = 0``").
V543_LAMBDA_STRONG: Final[float] = 0.0
V543_LAMBDA_DEAD: Final[float] = 0.0

#: PROCESS loss coefficients (:703) — the campaign YAML mirrors them.
V543_LAMBDA_RANK: Final[float] = 0.5
V543_LAMBDA_Q: Final[float] = 0.5
V543_LAMBDA_ENDPOINT: Final[float] = 0.012385049119773344

#: Ranking margin (:709 "``order_pairs`` 用与 V542 相同的 ``sample_order_pairs``").
V543_RANK_MARGIN: Final[float] = RANK_MARGIN_V53MP

#: The PROCESS frame-mask rule, quoted (plan :707, campaign YAML
#: ``tracks.process.process_frame_mask``).
PROCESS_FRAME_MASK_RULE: Final[str] = "endpoint_label_mask_all_statuses"
#: The MOVE frame-mask rule, quoted (plan :696 "只用 ``label_status == "ok"``").
MOVE_FRAME_MASK_RULE: Final[str] = "label_status_ok_only"

#: The frozen loss coefficient table the WP4 tests assert EXACTLY.
LOSS_COEFFICIENTS_V543: Final[Mapping[str, float]] = {
    "lambda_pair": V543_LAMBDA_PAIR,
    "lambda_cos": V543_LAMBDA_COS,
    "lambda_strong": V543_LAMBDA_STRONG,
    "lambda_dead": V543_LAMBDA_DEAD,
    "lambda_rank": V543_LAMBDA_RANK,
    "lambda_q": V543_LAMBDA_Q,
    "lambda_endpoint": V543_LAMBDA_ENDPOINT,
}

#: The plan §7.4 :750–767 resume-identity list, one entry per PLAN LINE (the
#: third line names three keys, so the flattened block has 16 entries).
RESUME_IDENTITY_GROUPS_V543: Final[tuple[tuple[str, ...], ...]] = (
    ("encoder_ckpt_sha256",),
    ("bridge_spec_sha256",),
    ("decoder_id", "track", "model_version"),
    ("contract_version",),
    ("seed",),
    ("config_sha256",),
    ("source_manifest_sha256",),
    ("data_manifest_sha256",),
    ("split_sha256",),
    ("label_spec_sha256",),
    ("sampler_spec_sha256",),
    ("environment_sha256",),
    ("code_fingerprint",),
    ("cache_identity",),
)

#: The flattened resume-identity key set (16 keys).
RESUME_IDENTITY_FIELDS_V543: Final[tuple[str, ...]] = tuple(
    key for group in RESUME_IDENTITY_GROUPS_V543 for key in group)

#: Run outcomes (six-word vocabulary of plan §7.4 / V541 §7.4).
RUN_OUTCOMES_V543: Final[tuple[str, ...]] = (
    "done", "scientific_failure", "invalid_run", "resource_failure",
    "budget_censored", "gate_skipped")

#: The skip reason an update records when the track has no valid supervision.
SKIPPED_NO_GRAD: Final[str] = "skipped_no_grad"


class V543TrainerError(V543Error):
    """Trainer input/state refusal (typed, fail-closed)."""

    def __init__(self, code: str, detail: str = "", *,
                 outcome: str = "invalid_run") -> None:
        super().__init__(code, detail)
        self.outcome = (outcome if outcome in RUN_OUTCOMES_V543
                        else "invalid_run")


# ---------------------------------------------------------------------------
# frozen-coefficient cross-check (a drift in V542 is a refusal, not a tuning)
# ---------------------------------------------------------------------------
def assert_frozen_loss_coefficients() -> dict[str, float]:
    """Refuse if the V5.4.2 frozen λ constants drifted from the V543 table.

    Plan §7.2 :688–711 says the V543 losses "原样复用 V542"; a difference in the
    V542 module would silently re-tune the campaign, so it is compared here
    (and re-checked in :meth:`V543TrainerConfig.__post_init__`).
    """
    pairs = {
        "lambda_rank": (V543_LAMBDA_RANK, V542_LAMBDA_RANK),
        "lambda_q": (V543_LAMBDA_Q, V542_LAMBDA_Q),
        "lambda_endpoint": (V543_LAMBDA_ENDPOINT, V542_LAMBDA_ENDPOINT),
    }
    drifted = {name: (ours, theirs) for name, (ours, theirs) in pairs.items()
               if float(ours) != float(theirs)}
    if drifted:
        raise V543TrainerError(
            "V543_LOSS_COEFFICIENT_DRIFT",
            "the V5.4.2 frozen λ constants no longer match the V5.4.3 table "
            f"(name: v543/v542) {drifted}; the V543 losses must reuse V542 "
            "verbatim (plan §7.2 :700–711)")
    config = KeyAtomLossV3Config()
    if float(config.lambda_pair) != V543_LAMBDA_PAIR:
        raise V543TrainerError(
            "V543_LOSS_COEFFICIENT_DRIFT",
            f"KeyAtomLossV3Config.lambda_pair is {config.lambda_pair}, "
            f"expected {V543_LAMBDA_PAIR} (plan §7.2 :691)")
    return dict(LOSS_COEFFICIENTS_V543)


def lr_at_update(update: int, *, warmup: int = V543_WARMUP_UPDATES,
                 peak: float = V543_LR) -> float:
    """Linear warmup then constant (plan §7.2 :660–661).

    ``update`` is 1-based: update 1 gets ``peak / warmup`` and update
    ``warmup + 1`` (and everything after) gets ``peak``.
    """
    step = max(int(update), 1)
    if step > int(warmup):
        return float(peak)
    return float(peak) * (float(step) / float(max(int(warmup), 1)))


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class V543TrainerConfig:
    """One planned V5.4.3 S1 run (plan §7.2 fixed schedule + campaign YAML)."""

    track: str = TRACK_MOVE
    decoder_id: str = "M0"
    seed: int = 17
    #: Flat dose of plan :659 (37,900).
    updates: int = V543_TOTAL_UPDATES
    warmup_updates: int = V543_WARMUP_UPDATES
    lr: float = V543_LR
    weight_decay: float = V543_WEIGHT_DECAY
    grad_clip: float = V543_GRAD_CLIP
    batch_curves: int = V543_BATCH_CURVES
    frames_per_curve: int = V543_FRAMES_PER_CURVE
    val_every: int = V543_VAL_EVERY
    inner_val_max_samples: int = V543_INNER_VAL_MAX_SAMPLES
    lambda_pair: float = V543_LAMBDA_PAIR
    lambda_cos: float = V543_LAMBDA_COS
    lambda_rank: float = V543_LAMBDA_RANK
    lambda_q: float = V543_LAMBDA_Q
    lambda_endpoint: float = V543_LAMBDA_ENDPOINT
    sampler: str = V543_SAMPLER_ID
    sampler_spec_sha256: str = V543_SAMPLER_SPEC_SHA256_PINNED
    precision: str = PRECISION_VERSION_V543
    device: str = "cpu"
    torch_compile: bool = False
    cache_mode: str = CACHE_MODE_FROZEN_ENCODER
    use_cache: bool = True
    cache_capacity: int = 200_000
    #: ``--stop-after N``: an honest smoke leg, never a formal-schedule run.
    stop_after: int | None = None
    checkpoint_every: int = V543_CHECKPOINT_EVERY
    log_every: int = 1
    order_min_gap: float = 0.05
    max_order_pairs_per_batch: int = 256
    feature_cache_capacity: int = 8_192
    require_identity_match: bool = True
    #: Descriptive label written into the logs / metrics (never used for control).
    run_tag: str = ""

    def __post_init__(self) -> None:
        if str(self.track) not in TRACKS_V543:
            raise V543TrainerError(
                "V543_TRACK", f"track must be one of {list(TRACKS_V543)}, got "
                f"{self.track!r}")
        if int(self.seed) not in V543_FROZEN_SEEDS:
            raise V543TrainerError(
                "V543_SEED_FROZEN",
                f"run seed must be one of {list(V543_FROZEN_SEEDS)} (plan "
                f"§7.2 :670), got {self.seed}")
        if int(self.updates) != V543_TOTAL_UPDATES:
            raise V543TrainerError(
                "V543_DOSE",
                f"updates must be the frozen {V543_TOTAL_UPDATES} (plan §7.2 "
                f":659: 37,900 flat); got {self.updates} — use ``stop_after`` "
                "for a smoke leg instead of reducing the dose")
        if int(self.warmup_updates) != V543_WARMUP_UPDATES:
            raise V543TrainerError(
                "V543_WARMUP",
                f"warmup must be {V543_WARMUP_UPDATES} updates (plan §7.2 "
                f":660), got {self.warmup_updates}")
        if float(self.lr) != V543_LR or float(self.weight_decay) != \
                V543_WEIGHT_DECAY or float(self.grad_clip) != V543_GRAD_CLIP:
            raise V543TrainerError(
                "V543_OPTIMIZER_FROZEN",
                "lr/weight_decay/grad_clip are frozen at "
                f"{V543_LR}/{V543_WEIGHT_DECAY}/{V543_GRAD_CLIP} (plan §7.2 "
                f":661–664), got {self.lr}/{self.weight_decay}/"
                f"{self.grad_clip}")
        if (int(self.batch_curves) != V543_BATCH_CURVES
                or int(self.frames_per_curve) != V543_FRAMES_PER_CURVE):
            raise V543TrainerError(
                "V543_BATCH_FROZEN",
                f"the batch is frozen at {V543_BATCH_CURVES} curves × "
                f"{V543_FRAMES_PER_CURVE} frames (plan §7.2 :666); got "
                f"{self.batch_curves}×{self.frames_per_curve}")
        if int(self.val_every) != V543_VAL_EVERY:
            raise V543TrainerError(
                "V543_VAL_EVERY",
                f"validation cadence is frozen at {V543_VAL_EVERY} updates "
                f"(plan §7.2 :668), got {self.val_every}")
        if int(self.inner_val_max_samples) != V543_INNER_VAL_MAX_SAMPLES:
            raise V543TrainerError(
                "V543_INNER_VAL_CAP",
                f"the inner-val cap is frozen at "
                f"{V543_INNER_VAL_MAX_SAMPLES} (plan §7.2 :668), got "
                f"{self.inner_val_max_samples}")
        if str(self.precision) != PRECISION_VERSION_V543:
            raise V543TrainerError(
                "V543_PRECISION",
                f"precision is frozen at {PRECISION_VERSION_V543!r} (plan "
                f"§7.2 :665), got {self.precision!r}")
        if bool(self.torch_compile):
            raise V543TrainerError(
                "V543_TORCH_COMPILE",
                "torch.compile is forbidden for the whole campaign (plan §7.2 "
                ":672, §11.3 禁止)")
        if str(self.sampler) != V543_SAMPLER_ID:
            raise V543TrainerError(
                "V543_SAMPLER_DRIFT",
                f"sampler must be {V543_SAMPLER_ID!r} (plan §7.2 :667), got "
                f"{self.sampler!r}")
        if str(self.sampler_spec_sha256) != V543_SAMPLER_SPEC_SHA256_PINNED:
            raise V543TrainerError(
                "V543_SAMPLER_DRIFT",
                "the pinned sampler spec SHA does not match the frozen digest "
                f"{V543_SAMPLER_SPEC_SHA256_PINNED} (plan §7.2 :667); a "
                "drifted spec is a typed refusal, never a silent re-draw")
        live_spec = sampler_spec_sha256()
        if live_spec != str(self.sampler_spec_sha256):
            raise V543TrainerError(
                "V543_SAMPLER_DRIFT",
                f"the LIVE sampler spec SHA {live_spec} differs from the "
                f"pinned {self.sampler_spec_sha256} — the sampler implementation "
                "drifted (plan §7.2 :667)")
        pairs = {
            "lambda_pair": (self.lambda_pair, V543_LAMBDA_PAIR),
            "lambda_cos": (self.lambda_cos, V543_LAMBDA_COS),
            "lambda_rank": (self.lambda_rank, V543_LAMBDA_RANK),
            "lambda_q": (self.lambda_q, V543_LAMBDA_Q),
            "lambda_endpoint": (self.lambda_endpoint, V543_LAMBDA_ENDPOINT),
        }
        drifted = {name: (ours, frozen) for name, (ours, frozen) in pairs.items()
                   if float(ours) != float(frozen)}
        if drifted:
            raise V543TrainerError(
                "V543_LOSS_FROZEN",
                f"loss coefficients are frozen (plan §7.2 :691/:703); drifted: "
                f"{drifted}")
        if self.stop_after is not None and int(self.stop_after) < 1:
            raise V543TrainerError(
                "V543_STOP_AFTER",
                f"stop_after must be >= 1, got {self.stop_after}")
        if int(self.checkpoint_every) < 1:
            raise V543TrainerError(
                "V543_CHECKPOINT_EVERY",
                f"checkpoint_every must be >= 1, got {self.checkpoint_every}")
        if str(self.cache_mode) not in (CACHE_MODE_FROZEN_ENCODER,
                                        "joint_finetune"):
            raise V543TrainerError(
                "V543_CACHE_MODE",
                f"cache_mode must be 'frozen_encoder_only' or "
                f"'joint_finetune', got {self.cache_mode!r}")

    # -- derived ------------------------------------------------------------
    @property
    def total_updates(self) -> int:
        """Updates THIS run executes (``stop_after`` or the flat dose)."""
        return (int(self.stop_after) if self.stop_after is not None
                else int(self.updates))

    @property
    def is_formal_schedule(self) -> bool:
        """True only for the frozen 37,900-update run with no smoke override."""
        return (self.stop_after is None
                and int(self.updates) == V543_TOTAL_UPDATES)

    @property
    def cache_enabled(self) -> bool:
        return bool(self.use_cache
                    and self.cache_mode == CACHE_MODE_FROZEN_ENCODER)

    def identity_dict(self) -> dict[str, Any]:
        """Config fields that change WHAT one update computes.

        The run-length / wall-clock knobs (``stop_after``, ``log_every``,
        ``run_tag``, cache sizing) stay OUT so a 3-update smoke leg can resume
        into a 6-update leg, while every scientific knob stays IN.
        """
        blob = asdict(self)
        for key in ("stop_after", "log_every", "run_tag", "cache_capacity",
                    "feature_cache_capacity", "device"):
            blob.pop(key, None)
        return blob

    def config_sha256(self) -> str:
        return canonical_sha256(self.identity_dict())

    def describe(self) -> dict[str, Any]:
        return {
            "schema": V543_TRAIN_SCHEMA,
            "track": self.track,
            "decoder_id": self.decoder_id,
            "seed": int(self.seed),
            "updates": int(self.updates),
            "total_updates_this_run": self.total_updates,
            "is_formal_schedule": self.is_formal_schedule,
            "warmup_updates": int(self.warmup_updates),
            "lr": float(self.lr),
            "weight_decay": float(self.weight_decay),
            "grad_clip": float(self.grad_clip),
            "batch": f"{self.batch_curves}x{self.frames_per_curve}",
            "val_every": int(self.val_every),
            "inner_val_max_samples": int(self.inner_val_max_samples),
            "precision": self.precision,
            "sampler": {"id": self.sampler,
                        "spec_sha256": self.sampler_spec_sha256},
            "loss_coefficients": dict(LOSS_COEFFICIENTS_V543),
            "frame_masks": {"move": MOVE_FRAME_MASK_RULE,
                            "process": PROCESS_FRAME_MASK_RULE},
            "cache": {"enabled": self.cache_enabled,
                      "mode": self.cache_mode,
                      "capacity": int(self.cache_capacity)},
            "torch_compile": bool(self.torch_compile),
        }


# ---------------------------------------------------------------------------
# one batch of training data (positions + collated tensors + audit metadata)
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class TrainBatch:
    """One collated batch plus everything the trainer needs OUTSIDE the model.

    ``batch`` carries supervision (``y_teacher`` / ``pair_index`` /
    ``pair_edit`` / ``frame_weight``); ``forward_view`` strips it, so the model
    path can never see a label (plan §7.2 :684).
    """

    positions: tuple[int, ...]
    batch: Mapping[str, Any]
    sample_ids: tuple[str, ...]
    curve_ids: tuple[str, ...]
    label_status: tuple[str, ...]
    split: str
    frame_keys: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        n_graphs = len(self.sample_ids)
        if len(self.positions) != n_graphs or len(self.curve_ids) != n_graphs \
                or len(self.label_status) != n_graphs:
            raise V543TrainerError(
                "V543_BATCH_METADATA",
                f"batch metadata is inconsistent: {len(self.positions)} "
                f"positions, {len(self.sample_ids)} sample ids, "
                f"{len(self.curve_ids)} curve ids, {len(self.label_status)} "
                "label statuses")
        if self.frame_keys and len(self.frame_keys) != n_graphs:
            raise V543TrainerError(
                "V543_BATCH_METADATA",
                f"{len(self.frame_keys)} frame cache keys for {n_graphs} "
                "graphs")

    @property
    def n_graphs(self) -> int:
        return len(self.sample_ids)

    @property
    def forward_view(self) -> dict[str, torch.Tensor]:
        """``forward_view_v3(batch)`` — supervision stripped (plan §7.2 :677)."""
        return forward_view_v3(dict(self.batch))


# ---------------------------------------------------------------------------
# data sources
# ---------------------------------------------------------------------------
class V543DataSource:
    """The trainer's data boundary (one implementable surface).

    Implementations must be DETERMINISTIC in ``update`` (plan §7.2 :667 /
    §7.4 :773) and must report the four data-side identity SHAs of §7.4.
    """

    n_train: int = 0
    n_inner_val: int = 0

    def draw_train_batch(self, update: int) -> TrainBatch:  # pragma: no cover
        raise NotImplementedError

    def inner_val_batches(self, limit: int) -> list[TrainBatch]:  # pragma: no cover
        raise NotImplementedError

    def identity(self) -> dict[str, str]:  # pragma: no cover
        raise NotImplementedError


class RealDataSource(V543DataSource):
    """The frozen 50K columnar pool, driven exactly like the V5.3-MP trainer.

    The data path is the S0-probe / V542 trainer path verbatim: frozen
    ``v53mp_pool_split(seed)`` → the `v541_curve_stratified_v1` sampler
    (``curve_batch_positions_v541`` with ``batch_rng_v541(seed, update)``) →
    ``_batch_from_positions`` over the frozen ``_FeatureLRU`` (plan §7.2
    :676–677).
    """

    def __init__(
            self,
            dataset: Any,
            *,
            seed: int,
            pool_curve_ids: Sequence[str],
            cache_identity: CacheIdentity | None = None,
            device: str = "cpu",
            feature_cache_capacity: int = 8_192,
            split_curve_ids: Mapping[str, str] | None = None,
    ) -> None:
        self.dataset = dataset
        self.seed = int(seed)
        self.device = str(device)
        self._identity = cache_identity
        train_pos, val_pos = v53mp_pool_split(
            dataset.samples, pool_curve_ids=pool_curve_ids, seed=self.seed)
        self._train_positions = [int(p) for p in train_pos]
        self._val_positions = [int(p) for p in val_pos]
        self._lru = _FeatureLRU(dataset, capacity=int(feature_cache_capacity))
        self._index = StrataIndex.build(
            dataset.samples, self._train_positions, dataset=dataset)
        self.n_train = len(self._train_positions)
        self.n_inner_val = len(self._val_positions)
        self._split_of_curve = dict(split_curve_ids or {})
        self._counter: dict[str, int] = {"forward": 0, "reverse": 0,
                                         "total": 0}

    # -- batch construction -------------------------------------------------
    def _build(self, positions: Sequence[int], split: str) -> TrainBatch:
        positions = [int(p) for p in positions]
        batch, records = _batch_from_positions(
            self.dataset, self._lru, positions, self.device)
        sample_ids = tuple(str(s) for s in batch["sample_ids"])
        curve_ids = tuple(str(c) for c in batch["curve_ids"])
        statuses = tuple(str(getattr(record, "label_status", ""))
                         for record in records)
        keys: tuple[str, ...] = ()
        if self._identity is not None:
            keys = tuple(
                self._frame_key(position, intent_id, curve_id, d0, split)
                for position, intent_id, curve_id, d0 in zip(
                    positions, batch["intent_ids"], curve_ids, batch["d0"]))
        return TrainBatch(
            positions=tuple(positions), batch=batch, sample_ids=sample_ids,
            curve_ids=curve_ids, label_status=statuses, split=split,
            frame_keys=keys)

    def _frame_key(self, position: int, intent_id: Any, curve_id: str,
                   d0: Any, split: str) -> str:
        sample = self.dataset.samples[int(position)]
        return frame_cache_key(
            self._identity, curve_id=str(curve_id),
            node_index=int(getattr(sample, "node_index", -1)),
            intent_id=intent_id, d0=float(d0), split=split)

    # -- V543DataSource -----------------------------------------------------
    def draw_train_batch(self, update: int) -> TrainBatch:
        rng = batch_rng_v541(self.seed, int(update))
        positions = curve_batch_positions_v541(
            self._index, rng=rng, batch_curves=V543_BATCH_CURVES,
            counters=self._counter)
        return self._build(positions, "inner_train")

    def inner_val_batches(self, limit: int) -> list[TrainBatch]:
        picks = self._val_positions[:max(int(limit), 0)]
        batches: list[TrainBatch] = []
        for start in range(0, len(picks), V543_BATCH_FRAMES):
            batches.append(self._build(
                picks[start:start + V543_BATCH_FRAMES], "inner_val"))
        return batches

    def identity(self) -> dict[str, str]:
        """The four data-side §7.4 SHAs (measured from the frozen artifacts)."""
        return {
            "source_manifest_sha256": _sha_of_manifest(
                getattr(self.dataset, "tables_dir", None),
                "teacher_manifest.json"),
            "data_manifest_sha256": _sha_of_manifest(
                getattr(self.dataset, "tables_dir", None),
                "data_manifest.json"),
            "split_sha256": _sha_of_manifest(
                getattr(self.dataset, "tables_dir", None),
                "split_manifest_v53mp.json"),
            "label_spec_sha256": sha256_text(
                "v543_labels|statuses=ok,masked,terminal,conflict|"
                "move=ok|process=ok,masked,terminal,conflict"),
        }

    # -- cache identity (plan §5.7 :443–450) -------------------------------
    @property
    def provides_cache_keys(self) -> bool:
        """True when this source can produce per-frame §5.7 cache keys."""
        return self._identity is not None

    @property
    def cache_encoder_sha256(self) -> str | None:
        """The encoder digest this source bakes into its frame keys.

        The trainer compares it with the encoder it actually holds; a mismatch
        would make every §5.7 key unique and the hit rate silently 0 % (a
        performance bug that looks like correct numerics), so it is a typed
        refusal instead.
        """
        return None if self._identity is None else \
            str(self._identity.encoder_ckpt_sha256)


def _sha_of_manifest(tables_dir: Any, name: str) -> str:
    """SHA-256 of a frozen data artifact, or a stable ``absent:`` marker."""
    if tables_dir is None:
        return f"absent:{name}"
    path = Path(str(tables_dir)) / name
    if not path.is_file():
        return f"absent:{name}"
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# supervision (the SEPARATE code path of plan §7.2 :681–684)
# ---------------------------------------------------------------------------
def process_frame_mask_v543(
        sample_ids: Sequence[str],
        endpoint_labels: Any,
        device: str,
) -> dict[str, Any]:
    """The PROCESS frame mask — endpoint-label validity over ALL statuses.

    Plan §7.2 :707 verbatim: "PROCESS 的 frame mask 不能沿用 ``_frame_mask_for``：
    必须使用 endpoint 标签表的有效 mask，覆盖 ``ok/terminal/masked/conflict``；
    terminal 行只要 endpoint 标签有效就进入训练".

    The mask is the FROZEN ``_endpoint_for_batch`` ``endpoint_mask`` —
    ``endpoint_label ∈ {0, 1}`` — so no second definition of "valid" is
    invented.  Terminal frames with a valid label are IN; ``_frame_mask_for``
    (``label_status == "ok"``) is deliberately NOT consulted here.
    """
    endpoint = _endpoint_for_batch(list(sample_ids), endpoint_labels,
                                  str(device))
    return {
        "process_frame_mask": endpoint["endpoint_mask"],
        "y_endpoint": endpoint["y_endpoint"],
        "endpoint_mask": endpoint["endpoint_mask"],
        "ts_row": endpoint["ts_row"],
        "start_row": endpoint["start_row"],
        "rule": PROCESS_FRAME_MASK_RULE,
    }


def move_frame_mask_v543(label_status: Sequence[str], device: str
                         ) -> dict[str, Any]:
    """The MOVE frame mask — ``label_status == "ok"`` only (plan :696)."""
    records = [_StatusRecord(status) for status in label_status]
    return {"move_frame_mask": _frame_mask_for(records, str(device)),
            "rule": MOVE_FRAME_MASK_RULE}


class _StatusRecord:
    """Minimal record carrying ``label_status`` (what ``_frame_mask_for`` reads)."""

    __slots__ = ("label_status",)

    def __init__(self, label_status: str) -> None:
        self.label_status = str(label_status)


def _supervision_for_batch(
        tb: TrainBatch,
        labels_v52: Any,
        endpoint_labels: Any,
        config: V543TrainerConfig,
        *,
        order_pairs: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Everything the losses need that is NOT the model output or context."""
    device = config.device
    lab = _labels_for_batch(list(tb.sample_ids), labels_v52, device)
    process = process_frame_mask_v543(tb.sample_ids, endpoint_labels, device)
    move = move_frame_mask_v543(tb.label_status, device)
    process_mask = process["process_frame_mask"]
    keep = process_mask.to(dtype=torch.bool)
    q_mask = lab["q_mask"] & keep
    pairs = order_pairs
    if pairs is not None and int(pairs.numel()):
        both = (keep[pairs[0].long()] & keep[pairs[1].long()])
        pairs = pairs[:, both]
    return {
        "q_true": lab["q_true"],
        "q_mask": q_mask,
        "q_mask_raw": lab["q_mask"],
        "process_frame_mask": process_mask,
        "y_endpoint": process["y_endpoint"],
        "endpoint_mask": process["endpoint_mask"],
        "ts_row": process["ts_row"],
        "start_row": process["start_row"],
        "move_frame_mask": move["move_frame_mask"],
        "label_status": tb.label_status,
        "order_pairs": pairs,
        "frame_weight": tb.batch["frame_weight"],
    }


# ---------------------------------------------------------------------------
# the two frozen loss assemblies (plan §7.2 :688–711)
# ---------------------------------------------------------------------------
def move_track_loss(
        out: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
        supervision: Mapping[str, Any],
        config: V543TrainerConfig,
) -> dict[str, Any]:
    """``L_M = L_atom + 0.5 * L_pair + 0.02 * L_cos`` by the FROZEN function.

    The MOVE loss is the V5.4.2 R3 total, obtained by calling
    ``key_atom_loss_v53mp`` with every PROCESS λ at 0 (plan :694–698 "复用
    ``key_atom_loss_v53mp`` / ``key_atom_loss_v3`` 的现有函数与 mask").  The
    ``u`` handed in is ``u0 + δu`` in DATASET key-row order (§4.3 :222).
    """
    result = key_atom_loss_v53mp(
        out["u"], batch["y_teacher"],
        batch[NODE_DEGK_KEY][batch[NODE_KEY_SET_KEY]],
        _key_ptr(batch), batch[NODE_POS_KEY], batch[NODE_KEY_SET_KEY],
        batch["pair_index"],
        lambda_rank=0.0, lambda_q=0.0, lambda_endpoint=0.0,
        config=KeyAtomLossV3Config(),
        lambda_cos=float(config.lambda_cos),
        frame_mask=supervision["move_frame_mask"],
        frame_weight=supervision["frame_weight"],
        pair_edit=batch["pair_edit"],
        lambda_strong=V543_LAMBDA_STRONG,
        lambda_dead=V543_LAMBDA_DEAD,
    )
    return {
        "loss": result["loss"],
        "l_move": result["l_move"],
        "l_atom": result["move"]["l_atom"],
        "l_pair": result["move"]["l_pair"],
        "l_cos": result["move"]["l_cos"],
        "n_frames": int(result["move"]["n_frames"]),
        "lambda_pair": float(result["move"]["lambda_pair"]),
        "lambda_cos": float(result["move"]["lambda_cos"]),
        "lambda_strong": float(result["move"]["lambda_strong"]),
        "lambda_dead": float(result["move"]["lambda_dead"]),
        "track": TRACK_MOVE,
        "schema": result["schema"],
    }


def process_track_loss(
        out: Mapping[str, torch.Tensor],
        batch: Mapping[str, Any],
        supervision: Mapping[str, Any],
        config: V543TrainerConfig,
        *,
        endpoint_pos_weight: float,
) -> dict[str, Any]:
    """``L_P = 0.5*L_rank + 0.5*L_q + 0.012385049119773344*L_endpoint``.

    Plan §7.2 :705–709: "直接调用 ``ranking_loss_v53mp``、``progress_loss_v52``、
    ``endpoint_loss_v53mp``，不调用 MOVE 损失".  The row gate is
    :func:`process_frame_mask_v543` (endpoint-label validity over every
    ``label_status``), NOT ``_frame_mask_for`` — the F5 correction.
    """
    pairs = supervision["order_pairs"]
    if pairs is None:
        pairs = torch.zeros((2, 0), dtype=torch.long)
    rank = ranking_loss_v53mp(
        out["q"], pairs, margin=V543_RANK_MARGIN,
        frame_weight=supervision["frame_weight"])
    q_out = progress_loss_v52(
        out["q"], supervision["q_true"], supervision["q_mask"],
        frame_weight=supervision["frame_weight"])
    endpoint = endpoint_loss_v53mp(
        out["endpoint_logit"], supervision["y_endpoint"],
        supervision["endpoint_mask"],
        endpoint_pos_weight=float(endpoint_pos_weight),
        frame_weight=supervision["frame_weight"])
    loss = (V543_LAMBDA_RANK * rank["l_rank"]
            + V543_LAMBDA_Q * q_out["l_q"]
            + V543_LAMBDA_ENDPOINT * endpoint["l_endpoint"])
    return {
        "loss": loss,
        "l_rank": rank["l_rank"],
        "l_q": q_out["l_q"],
        "l_endpoint": endpoint["l_endpoint"],
        "n_rank_pairs": int(rank["n_rank_pairs"]),
        "n_q_rows": int(q_out["n_q_rows"]),
        "n_endpoint_rows": int(endpoint["n_endpoint_rows"]),
        "endpoint_pos_weight": float(endpoint_pos_weight),
        "process_mask_rows": int(
            supervision["process_frame_mask"].sum().item()),
        "lambda_rank": V543_LAMBDA_RANK,
        "lambda_q": V543_LAMBDA_Q,
        "lambda_endpoint": V543_LAMBDA_ENDPOINT,
        "track": TRACK_PROCESS,
        "schema": "pesn_v543_process_loss_v1",
    }


def track_has_supervision(supervision: Mapping[str, Any], track: str) -> bool:
    """False ⇒ the update records ``skipped_no_grad`` (plan §7.2 :711)."""
    if track == TRACK_MOVE:
        return bool(supervision["move_frame_mask"].any())
    return bool(supervision["process_frame_mask"].any())


# ---------------------------------------------------------------------------
# identity (§7.4 :750–772)
# ---------------------------------------------------------------------------
#: Files whose bytes define the V5.4.3 code fingerprint (§7.4 :765).
CODE_FINGERPRINT_FILES_V543: Final[tuple[str, ...]] = (
    "pes2ts_core/pesn/models/v543/contracts.py",
    "pes2ts_core/pesn/models/v543/coarse_encoder.py",
    "pes2ts_core/pesn/models/v543/latent_bridge.py",
    "pes2ts_core/pesn/models/v543/registry.py",
    "pes2ts_core/pesn/models/v543/assembly.py",
    "pes2ts_core/pesn/models/v543/model.py",
    "pes2ts_core/pesn/offline/data/v543_feature_cache.py",
    "pes2ts_core/pesn/offline/training/v543_train.py",
)


def code_fingerprint_v543(root: str | os.PathLike[str] | None = None) -> str:
    """Digest over the V5.4.3 module bytes (the §7.4 ``code_fingerprint``).

    Deterministic and git-independent: a source edit changes it, which makes a
    resume across an edited trainer a typed ``invalid_run`` instead of a
    silently different run (§7.4 :771).
    """
    base = Path(root) if root is not None else Path(__file__).resolve().parents[4]
    import hashlib

    digest = hashlib.sha256()
    for name in CODE_FINGERPRINT_FILES_V543:
        path = base / name
        digest.update(name.encode("utf-8"))
        if path.is_file():
            digest.update(path.read_bytes())
        else:                                  # pragma: no cover - moved file
            digest.update(b"absent")
    return digest.hexdigest()


def encoder_digest_v543(encoder: Any) -> str:
    """The encoder identity of the §7.4 ``encoder_ckpt_sha256`` slot.

    The strict-load audit's digest when the encoder was LOADED (the S0-A asset),
    otherwise the digest of the freshly initialised ``state_dict()`` — a real
    measurement either way, never an empty placeholder.
    """
    audit = getattr(encoder, "load_audit", None)
    if isinstance(audit, Mapping):
        recorded = audit.get("encoder_ckpt_sha256")
        if isinstance(recorded, str) and recorded:
            return recorded
    from pes2ts_core.pesn.models.v542.identity import state_dict_sha256

    return state_dict_sha256(encoder.state_dict())


def identity_block_v543(
        *,
        model: V543Model,
        config: V543TrainerConfig,
        source: V543DataSource,
        cache: PacketCache | None,
        encoder_ckpt_sha256: str,
) -> dict[str, Any]:
    """The 16-key resume identity (§7.4 :750–767, 14 plan lines).

    Every value is MEASURED (the encoder digest, the Bridge spec digest, the
    live sampler spec SHA, the data artifacts' SHAs, the environment block, the
    code fingerprint) or declared by the run config; there is no "unknown"
    slot, so a resume comparison is always decisive.
    """
    assembly = model.assembly
    data_identity = source.identity()
    environment = canonical_sha256(env_state())
    block: dict[str, Any] = {
        "encoder_ckpt_sha256": str(encoder_ckpt_sha256),
        "bridge_spec_sha256": str(assembly.bridge.bridge_spec_sha256()),
        "decoder_id": str(assembly.active.decoder_id),
        "track": str(assembly.active_track),
        "model_version": str(assembly.active.model_version),
        "contract_version": DECODER_CONTEXT_SCHEMA_V543,
        "seed": int(config.seed),
        "config_sha256": config.config_sha256(),
        "source_manifest_sha256": str(
            data_identity.get("source_manifest_sha256", "")),
        "data_manifest_sha256": str(
            data_identity.get("data_manifest_sha256", "")),
        "split_sha256": str(data_identity.get("split_sha256", "")),
        "label_spec_sha256": str(data_identity.get("label_spec_sha256", "")),
        "sampler_spec_sha256": sampler_spec_sha256(),
        "environment_sha256": environment,
        "code_fingerprint": code_fingerprint_v543(),
        "cache_identity": ("disabled" if cache is None
                           else canonical_sha256(cache.identity.describe())),
    }
    missing = [key for key in RESUME_IDENTITY_FIELDS_V543
               if key not in block]
    if missing:                                # pragma: no cover - guard
        raise V543TrainerError(
            "V543_IDENTITY_INCOMPLETE",
            f"the identity block is missing {missing} (plan §7.4 :750–767)")
    return block


def assert_resume_identity_v543(
        state: Mapping[str, Any],
        expected: Mapping[str, Any],
) -> dict[str, Any]:
    """Typed refusal unless ``checkpoint_last`` matches THIS identity.

    Plan §7.4 :771: "恢复只从 ``checkpoint_last.pt``，且 identity 逐键 + digest
    匹配；不匹配 = ``invalid_run``，不得静默新建".  Both checks are made: the
    key-by-key comparison (offending keys land on ``exc.fields``-style detail)
    and the canonical digest of the block.
    """
    if not isinstance(state, Mapping):
        raise V543TrainerError(
            "V543_RESUME_INVALID",
            f"checkpoint payload must be a mapping, got "
            f"{type(state).__name__}")
    if state.get("schema") != V543_TRAIN_SCHEMA:
        raise V543TrainerError(
            "V543_RESUME_IDENTITY",
            f"checkpoint schema {state.get('schema')!r} != "
            f"{V543_TRAIN_SCHEMA!r}")
    stored = state.get("identity")
    if not isinstance(stored, Mapping):
        raise V543TrainerError(
            "V543_RESUME_IDENTITY",
            "checkpoint_last carries no identity block; resume would be a "
            "silent fresh start (plan §7.4 :771)")
    stored = dict(stored)
    expected = {key: expected[key] for key in RESUME_IDENTITY_FIELDS_V543}
    sentinel = object()
    mismatches = sorted(
        key for key in set(expected) | set(stored)
        if expected.get(key, sentinel) != stored.get(key, sentinel))
    if mismatches:
        detail = "; ".join(
            f"{key}: checkpoint={stored.get(key, '<absent>')!r} "
            f"run={expected.get(key, '<absent>')!r}" for key in mismatches)
        raise V543TrainerError(
            "V543_RESUME_IDENTITY",
            "checkpoint_last belongs to a different run — " + detail)
    digest = canonical_sha256(expected)
    if stored.get("identity_sha256") != digest and state.get(
            "identity_sha256") != digest:
        raise V543TrainerError(
            "V543_RESUME_IDENTITY",
            f"identity digest mismatch (stored="
            f"{state.get('identity_sha256')!r}, expected={digest!r})")
    return stored


# ---------------------------------------------------------------------------
# the trainer
# ---------------------------------------------------------------------------
class V543Trainer:
    """Single-track, decoder-only S1 trainer (plan §7.2)."""

    def __init__(
            self,
            *,
            model: V543Model,
            config: V543TrainerConfig,
            source: V543DataSource,
            labels_v52: Any,
            endpoint_labels: Any,
            out_dir: str | os.PathLike[str],
            cache: PacketCache | None = None,
            endpoint_pos_weight: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(model, V543Model):
            raise V543TrainerError(
                "V543_MODEL", f"model must be a V543Model, got "
                f"{type(model).__name__}")
        if str(config.track) != str(model.active_track):
            raise V543TrainerError(
                "V543_TRACK_MISMATCH",
                f"config.track={config.track!r} but the model is assembled for "
                f"{model.active_track!r} (plan §4.4 :232)")
        if str(config.decoder_id) != str(model.assembly.active.decoder_id):
            raise V543TrainerError(
                "V543_DECODER_MISMATCH",
                f"config.decoder_id={config.decoder_id!r} but the assembly "
                f"holds {model.assembly.active.decoder_id!r}")
        assert_frozen_loss_coefficients()
        self.model = model
        self.config = config
        self.source = source
        self.labels_v52 = labels_v52
        self.endpoint_labels = endpoint_labels
        self.out_dir = Path(out_dir)
        self.assembly: V543Assembly = model.assembly
        self.assembly.assert_encoder_frozen()
        self._assert_fp32()
        self._params = list(model.trainable_parameters())
        self.optimizer: torch.optim.Optimizer | None = None
        if self._params:
            self.optimizer = torch.optim.AdamW(
                self._params, lr=float(config.lr),
                weight_decay=float(config.weight_decay))
        self.encoder_ckpt_sha256 = encoder_digest_v543(self.assembly.encoder)
        self.cache = cache
        if config.cache_enabled and self.cache is None:
            self.cache = PacketCache(
                CacheIdentity(
                    encoder_ckpt_sha256=self.encoder_ckpt_sha256,
                    bridge_spec_sha256=str(
                        self.assembly.bridge.bridge_spec_sha256()),
                    precision_version=PRECISION_VERSION_V543,
                    mode=CACHE_MODE_FROZEN_ENCODER),
                capacity=int(config.cache_capacity))
        if not config.cache_enabled:
            self.cache = None
        self._assert_cache_usable()
        self.pos_weight_info = dict(
            endpoint_pos_weight or self._estimate_endpoint_pos_weight())
        self.endpoint_pos_weight = float(
            self.pos_weight_info.get("pos_weight", 1.0))
        self.identity = identity_block_v543(
            model=self.model, config=self.config, source=self.source,
            cache=self.cache, encoder_ckpt_sha256=self.encoder_ckpt_sha256)
        self.identity_sha256 = canonical_sha256(self.identity)
        self._update = 0
        self._best_val = math.inf
        self._best_update = 0
        self._skipped = 0
        self._rng = rng_state_snapshot("cpu")
        self._val_history: list[dict[str, Any]] = []
        self._train_rows: list[dict[str, Any]] = []

    # -- preparation --------------------------------------------------------
    def _assert_cache_usable(self) -> None:
        """Fail closed when the cache could only ever MISS (plan §5.7 :457).

        Two silent-performance traps are turned into typed refusals:

        * the data source cannot produce per-frame §5.7 keys at all
          (``V543_CACHE_KEYS_MISSING`` — the cache would be dead weight);
        * the source bakes a DIFFERENT encoder digest into its frame keys than
          the encoder this trainer holds (``V543_CACHE_IDENTITY`` — every key
          would be unique and the hit rate would be silently 0 %).
        """
        if self.cache is None:
            return
        provides = getattr(self.source, "provides_cache_keys", None)
        if provides is False:
            raise V543TrainerError(
                "V543_CACHE_KEYS_MISSING",
                "the cache is enabled but the data source provides no "
                "per-frame §5.7 cache keys (pass a CacheIdentity to the "
                "source, or run with use_cache=False)")
        source_sha = getattr(self.source, "cache_encoder_sha256", None)
        if source_sha and str(source_sha) != str(self.encoder_ckpt_sha256):
            raise V543TrainerError(
                "V543_CACHE_IDENTITY",
                "the data source keys its frames with encoder_ckpt_sha256="
                f"{source_sha!r} but the trainer holds "
                f"{self.encoder_ckpt_sha256!r}; every lookup would miss "
                "(plan §5.7 :457)")

    def _assert_fp32(self) -> None:
        """The whole model is fp32 (plan §7.2 :665 precision row)."""
        offenders: list[str] = []
        modules = {
            "encoder": self.assembly.encoder,
            "bridge": self.assembly.bridge,
            "move": self.assembly.move.decoder,
            "process": self.assembly.process.decoder,
        }
        for label, module in modules.items():
            for name, parameter in getattr(module, "named_parameters",
                                           lambda: [])():
                if parameter.dtype != torch.float32:
                    offenders.append(f"{label}.{name}:{parameter.dtype}")
                if parameter.is_complex():
                    offenders.append(f"{label}.{name}:complex")
        if offenders:
            raise V543TrainerError(
                "V543_PRECISION",
                f"fp32 is required everywhere (plan §7.2 :665); offenders: "
                f"{offenders[:6]}")

    def _estimate_endpoint_pos_weight(self) -> dict[str, Any]:
        """``endpoint_pos_weight`` from inner-train ONLY, estimated ONCE (:708).

        Uses the frozen estimator over the inner-train positions of the frozen
        pool split; a provider without an endpoint table gets the neutral 1.0
        with the reason recorded (never a fabricated estimate).
        """
        from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
            estimate_endpoint_pos_weight,
        )

        positions = getattr(self.source, "_train_positions", None)
        samples = getattr(getattr(self.source, "dataset", None), "samples", None)
        if positions and samples is not None and self.endpoint_labels is not None:
            try:
                info = estimate_endpoint_pos_weight(
                    samples, self.endpoint_labels, positions,
                    seed=int(self.config.seed))
                info["estimated_by"] = "estimate_endpoint_pos_weight (frozen)"
                info["scope"] = "inner_train_only"
                return dict(info)
            except Exception as error:  # noqa: BLE001 - recorded, not hidden
                return {"pos_weight": 1.0, "n_pos": 0, "n_neg": 0,
                        "source": f"estimator unavailable ({type(error).__name__})",
                        "estimated_by": "fallback",
                        "scope": "inner_train_only"}
        return {"pos_weight": 1.0, "n_pos": 0, "n_neg": 0,
                "source": "provider carries no endpoint table (synthetic)",
                "estimated_by": "fallback", "scope": "inner_train_only"}

    def _prepare_out_dir(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # -- context acquisition (cache-aware, plan §7.3) -----------------------
    def _context_for(self, tb: TrainBatch) -> tuple[Any, str]:
        """Return ``(context, source)`` where source ∈ {full_hit, encode}."""
        view = tb.forward_view
        if self.cache is not None and tb.frame_keys:
            hits = self.cache.lookup_many(tb.frame_keys)
            if all(hit is not None for hit in hits):
                return (concat_contexts([hit.context for hit in hits]),
                        "full_hit")
        encoder = self.assembly.encoder
        before = int(encoder.forward_count)
        packet = encoder(view)
        calls = int(encoder.forward_count) - before
        if calls != 1:
            raise V543TrainerError(
                "V543_ENCODER_CALLS",
                f"one update must run the CoarseEncoder exactly once (plan "
                f"§4.4 :231), observed {calls}")
        context = self.assembly.bridge.build_context(packet, view)
        if self.cache is not None and tb.frame_keys:
            for key, part in zip(tb.frame_keys,
                                 split_context_by_graph(context)):
                self.cache.put(key, FrameRecord(context=part))
            self.cache.note_builds(1)
        return context, "encode"

    # -- one update ---------------------------------------------------------
    def _order_pairs(self, tb: TrainBatch, update: int) -> torch.Tensor | None:
        if self.config.track != TRACK_PROCESS:
            return None
        return sample_order_pairs(
            tb.batch, self.labels_v52,
            order_min_gap=float(self.config.order_min_gap),
            max_pairs=int(self.config.max_order_pairs_per_batch),
            rng=_update_rng(int(self.config.seed), int(update), salt=2))

    def _loss_on_batch(self, tb: TrainBatch, update: int
                       ) -> tuple[dict[str, Any], dict[str, Any], str, bool]:
        """Forward + frozen loss for ONE batch (no backward)."""
        supervision = _supervision_for_batch(
            tb, self.labels_v52, self.endpoint_labels, self.config,
            order_pairs=self._order_pairs(tb, update))
        if not track_has_supervision(supervision, self.config.track):
            return {}, supervision, "no_mask", True
        context, origin = self._context_for(tb)
        out = self.model.forward_context(context)
        check_external_output(out)
        if self.config.track == TRACK_MOVE:
            loss_out = move_track_loss(out, tb.batch, supervision, self.config)
        else:
            loss_out = process_track_loss(
                out, tb.batch, supervision, self.config,
                endpoint_pos_weight=self.endpoint_pos_weight)
        if not bool(torch.isfinite(loss_out["loss"])):
            raise V543TrainerError(
                "V543_NONFINITE_LOSS",
                f"update {update}: the {self.config.track} loss is not finite")
        return loss_out, supervision, origin, False

    def _train_step(self, update: int, tb: TrainBatch) -> dict[str, Any]:
        started = time.perf_counter()
        if self.optimizer is not None:
            self.optimizer.zero_grad(set_to_none=True)
            loss_out, supervision, origin, skipped = self._loss_on_batch(
                tb, update)
            if skipped:
                self._skipped += 1
                return self._skipped_row(update, tb, "no_mask")
            loss_out["loss"].backward()
            isolation = self.model.assert_gradient_isolation(
                require_active_grad=bool(self._params))
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                self._params, float(self.config.grad_clip)))
            finite = bool(math.isfinite(grad_norm) and all(
                bool(torch.isfinite(parameter.grad).all())
                for parameter in self._params if parameter.grad is not None))
            lr = lr_at_update(update, warmup=self.config.warmup_updates,
                              peak=float(self.config.lr))
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self.optimizer.step()
        else:
            with torch.no_grad():
                loss_out, supervision, origin, skipped = self._loss_on_batch(
                    tb, update)
            if skipped:
                self._skipped += 1
                return self._skipped_row(update, tb, "no_mask")
            isolation = self.model.assert_gradient_isolation(
                require_active_grad=False)
            grad_norm = None
            finite = None
            lr = 0.0
        row = self._train_row(update, tb, loss_out, supervision, origin,
                              grad_norm=grad_norm, lr=lr,
                              seconds=time.perf_counter() - started)
        row["grad_finite"] = finite
        row["isolation"] = isolation
        return row

    def _skipped_row(self, update: int, tb: TrainBatch, reason: str
                     ) -> dict[str, Any]:
        return {
            "update": int(update),
            "event": SKIPPED_NO_GRAD,
            "reason": reason,
            "track": self.config.track,
            "decoder_id": self.config.decoder_id,
            "seed": int(self.config.seed),
            "n_graphs": int(tb.n_graphs),
            "split": tb.split,
            "label_status": list(tb.label_status),
        }

    def _train_row(self, update: int, tb: TrainBatch,
                   loss_out: Mapping[str, Any],
                   supervision: Mapping[str, Any], origin: str, *,
                   grad_norm: float | None, lr: float,
                   seconds: float) -> dict[str, Any]:
        row: dict[str, Any] = {
            "update": int(update),
            "event": "train",
            "track": self.config.track,
            "decoder_id": self.config.decoder_id,
            "seed": int(self.config.seed),
            "loss": _as_float(loss_out["loss"]),
            "lr": float(lr),
            "grad_norm": grad_norm,
            "seconds": round(float(seconds), 6),
            "n_graphs": int(tb.n_graphs),
            "context_source": origin,
            "n_process_mask_rows": int(
                supervision["process_frame_mask"].sum().item()),
            "n_move_mask_rows": int(
                supervision["move_frame_mask"].sum().item()),
            "label_status": list(tb.label_status),
            "encoder_forward_count": int(
                self.assembly.encoder.forward_count),
        }
        for key in ("l_atom", "l_pair", "l_cos", "l_move", "l_rank", "l_q",
                    "l_endpoint", "n_rank_pairs", "n_q_rows",
                    "n_endpoint_rows", "n_frames"):
            if key in loss_out:
                value = loss_out[key]
                row[key] = (int(value) if isinstance(value, int)
                            else _as_float(value))
        return row

    # -- validation ---------------------------------------------------------
    def validate(self, update: int) -> dict[str, Any]:
        """Inner-val loss over ≤8,192 frames (plan §7.2 :668)."""
        batches = self.source.inner_val_batches(
            int(self.config.inner_val_max_samples))
        total = 0.0
        weight_sum = 0.0
        skipped = 0
        components: dict[str, float] = {}
        with torch.no_grad():
            for index, tb in enumerate(batches):
                loss_out, supervision, _origin, was_skipped = \
                    self._loss_on_batch(tb, int(update) * 1000 + index)
                if was_skipped:
                    skipped += 1
                    continue
                weight = float(tb.batch["frame_weight"].sum().item())
                total += _as_float(loss_out["loss"]) * weight
                weight_sum += weight
                for key in ("l_atom", "l_pair", "l_cos", "l_move", "l_rank",
                            "l_q", "l_endpoint"):
                    if key in loss_out:
                        components[key] = components.get(key, 0.0) + \
                            _as_float(loss_out[key]) * weight
        value = total / weight_sum if weight_sum > 0.0 else math.inf
        row = {
            "update": int(update),
            "event": "validate",
            "track": self.config.track,
            "decoder_id": self.config.decoder_id,
            "seed": int(self.config.seed),
            "inner_val_loss": (None if not math.isfinite(value) else value),
            "n_batches": len(batches),
            "n_skipped_batches": int(skipped),
            "weight_sum": weight_sum,
            "components": ({key: value / weight_sum
                            for key, value in components.items()}
                           if weight_sum > 0.0 else {}),
            "cache": (None if self.cache is None else self.cache.stats()),
            "encoder_forward_count": int(
                self.assembly.encoder.forward_count),
        }
        self._val_history.append(row)
        return row

    # -- checkpoints --------------------------------------------------------
    def _checkpoint_payload(self, update: int, *, kind: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": V543_TRAIN_SCHEMA,
            "kind": str(kind),
            "update": int(update),
            "identity": dict(self.identity),
            "identity_sha256": self.identity_sha256,
            "config": self.config.describe(),
            "params": self.assembly.param_report(),
            "endpoint_pos_weight": dict(self.pos_weight_info),
            "best_val_loss": (None if not math.isfinite(self._best_val)
                              else float(self._best_val)),
            "best_update": int(self._best_update),
            "skipped_updates": int(self._skipped),
            "cache": (None if self.cache is None else self.cache.stats()),
            "model": _state_dict_of(self.assembly.active.decoder),
            "optimizer": (None if self.optimizer is None
                          else self.optimizer.state_dict()),
            "rng": rng_state_snapshot("cpu"),
            "sampler_counters": {
                str(k): int(v) for k, v in
                (getattr(self.source, "_counter", None) or {}).items()},
            "schedule": {
                "updates": int(self.config.updates),
                "warmup_updates": int(self.config.warmup_updates),
                "lr": float(self.config.lr),
                "weight_decay": float(self.config.weight_decay),
                "grad_clip": float(self.config.grad_clip),
                "val_every": int(self.config.val_every),
                "checkpoint_every": int(self.config.checkpoint_every),
            },
            "inactive": _state_dict_of(self.assembly.inactive.decoder),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        payload["content_digest_sha256"] = _payload_digest(payload)
        return payload

    def _save(self, update: int, *, kind: str) -> Path:
        self._prepare_out_dir()
        name = ("checkpoint_best.pt" if kind == "best"
                else "checkpoint_last.pt")
        path = self.out_dir / name
        payload = self._checkpoint_payload(update, kind=kind)
        _atomic_torch_save(path, payload)
        return path

    def resume_from(self, path: str | os.PathLike[str]) -> dict[str, Any]:
        """Restore from ``checkpoint_last.pt`` after an identity check (§7.4)."""
        source = Path(path)
        if not source.is_file():
            raise V543TrainerError(
                "V543_RESUME_MISSING",
                f"resume checkpoint not found: {source}")
        state = torch.load(str(source), map_location="cpu",
                           weights_only=False)
        stored = assert_resume_identity_v543(state, self.identity)
        self.assembly.active.decoder.load_state_dict(state["model"])
        if self.optimizer is not None and state.get("optimizer") is not None:
            self.optimizer.load_state_dict(state["optimizer"])
        self._update = int(state.get("update", 0))
        self._best_val = (math.inf if state.get("best_val_loss") is None
                          else float(state["best_val_loss"]))
        self._best_update = int(state.get("best_update", 0))
        self._skipped = int(state.get("skipped_updates", 0))
        if isinstance(state.get("rng"), Mapping):
            rng_state_restore(state["rng"])
        counters = state.get("sampler_counters")
        if isinstance(counters, Mapping) and hasattr(self.source, "_counter"):
            live = getattr(self.source, "_counter", None)
            if isinstance(live, dict):
                live.update({str(k): int(v) for k, v in counters.items()})
        return stored

    # -- the loop -----------------------------------------------------------
    def run(self) -> dict[str, Any]:
        """Run the configured updates and return the ``metrics.json`` payload."""
        started = time.perf_counter()
        self._prepare_out_dir()
        train_log = self.out_dir / "train_log.jsonl"
        run_records = self.out_dir / "run_records.jsonl"
        steps = 0
        for update in range(self._update + 1, self.config.total_updates + 1):
            tb = self.source.draw_train_batch(int(update))
            row = self._train_step(int(update), tb)
            self._update = int(update)
            steps += 1
            if self.config.log_every and update % int(self.config.log_every) == 0:
                _append_jsonl(train_log, [row])
            if update % int(self.config.val_every) == 0 or \
                    update == self.config.total_updates:
                val = self.validate(int(update))
                _append_jsonl(run_records, [val])
                if val["inner_val_loss"] is not None and \
                        float(val["inner_val_loss"]) < self._best_val:
                    self._best_val = float(val["inner_val_loss"])
                    self._best_update = int(update)
                    self._save(int(update), kind="best")
            if update % int(self.config.checkpoint_every) == 0 or \
                    update == self.config.total_updates:
                self._save(int(update), kind="last")
        metrics = self._metrics(steps, time.perf_counter() - started)
        _write_json(self.out_dir / "metrics.json", metrics)
        return metrics

    def _metrics(self, steps: int, seconds: float) -> dict[str, Any]:
        last_val = self._val_history[-1] if self._val_history else None
        return {
            "schema": V543_TRAIN_SCHEMA,
            "track": self.config.track,
            "decoder_id": self.config.decoder_id,
            "seed": int(self.config.seed),
            "outcome": "done",
            "reason_code": None,
            "is_formal_schedule": self.config.is_formal_schedule,
            "updates": int(self._update),
            "steps_this_call": int(steps),
            "skipped_updates": int(self._skipped),
            "wall_seconds": round(float(seconds), 3),
            "seconds_per_update": (round(float(seconds) / max(steps, 1), 6)),
            "best_val_loss": (None if not math.isfinite(self._best_val)
                              else float(self._best_val)),
            "best_update": int(self._best_update),
            "last_val": last_val,
            "identity": dict(self.identity),
            "identity_sha256": self.identity_sha256,
            "config": self.config.describe(),
            "params": self.assembly.param_report(),
            "endpoint_pos_weight": dict(self.pos_weight_info),
            "cache": (None if self.cache is None else self.cache.stats()),
            "encoder_forward_count": int(
                self.assembly.encoder.forward_count),
            "gradient_isolation": {
                "encoder_requires_grad": False,
                "inactive_requires_grad": False,
                "active_trainable_params": len(self._params),
                "rule": "plan §7.2 :713–726",
            },
            "loss_coefficients": dict(LOSS_COEFFICIENTS_V543),
            "frame_masks": {"move": MOVE_FRAME_MASK_RULE,
                            "process": PROCESS_FRAME_MASK_RULE},
        }


# ---------------------------------------------------------------------------
# small io / helper utilities
# ---------------------------------------------------------------------------
def _as_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().to("cpu").reshape(()).item())
    return float(value)


def _state_dict_of(module: Any) -> dict[str, torch.Tensor]:
    state = getattr(module, "state_dict", None)
    if not callable(state):
        return {}
    return {key: value.detach().cpu() for key, value in state().items()}


def _payload_digest(payload: Mapping[str, Any]) -> str:
    """Digest over the JSON-safe projection of a checkpoint payload.

    Tensor/array-bearing members (``model`` / ``optimizer`` / ``inactive``) are
    excluded — their identity is the ``state_dict`` load, not a digest — and
    every other value goes through ``json.dumps(default=str)`` so numpy scalars
    cannot break the record (the frozen ``canonical_sha256`` refuses non-JSON
    payloads by design).
    """
    import hashlib

    projection = {key: value for key, value in payload.items()
                  if key not in ("model", "optimizer", "inactive")}
    blob = json.dumps(projection, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(payload), str(tmp))
    os.replace(str(tmp), str(path))


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True, default=str)
                         + "\n")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True,
                               ensure_ascii=False, default=str),
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# module-level entry point (plan §11.2 ``train_run_v543``)
# ---------------------------------------------------------------------------
def train_run_v543(
        model: V543Model,
        config: V543TrainerConfig,
        source: V543DataSource,
        *,
        labels_v52: Any,
        endpoint_labels: Any,
        out_dir: str | os.PathLike[str],
        cache: PacketCache | None = None,
        resume_from: str | os.PathLike[str] | None = None,
        endpoint_pos_weight: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and run a :class:`V543Trainer` (plan §11.2 API)."""
    trainer = V543Trainer(
        model=model, config=config, source=source, labels_v52=labels_v52,
        endpoint_labels=endpoint_labels, out_dir=out_dir, cache=cache,
        endpoint_pos_weight=endpoint_pos_weight)
    if resume_from is not None:
        trainer.resume_from(resume_from)
    return trainer.run()


# ---------------------------------------------------------------------------
# CLI — the WP8 smoke gate entry point (``--stop-after N``)
# ---------------------------------------------------------------------------
def build_real_trainer(
        *,
        track: str,
        decoder_id: str,
        seed: int,
        out_dir: str | os.PathLike[str],
        stop_after: int | None = None,
        use_cache: bool = True,
        campaign_config: str | None = None,
        data_view_config: str | None = None,
        encoder_ckpt_sha256: str | None = None,
        val_batches: int | None = None,
) -> V543Trainer:
    """Wire the frozen real-data run (dataset → source → model → trainer).

    The data/asset discovery reuses the WP3 S0-probe helpers (the same pinned
    campaign assets and the same frozen data view), so a smoke run cannot
    disagree with the S0 certificates about which encoder or which split it
    used.  ``stop_after`` produces an HONEST short run: the dose stays 37,900
    in the config, ``is_formal_schedule`` is False, and the reason is recorded
    in ``metrics.json``.

    Refusals: every typed V543 trainer refusal, plus the yaml/asset errors of
    the pinned campaign files.
    """
    from pes2ts_core.pesn.models.v543.registry import TRACK_MOVE

    repo_root = Path(__file__).resolve().parents[4]
    campaign_path = campaign_config or (
        "config/campaigns/pes2ts_v543_dual_decoder_50k.yaml")
    data_path = data_view_config or (
        "config/campaigns/pes2ts_v542_backbone_50k.yaml")

    from scripts.agent_ops.v542_a0_parity import (
        _open_frozen_dataset,
        load_pinned_assets,
    )
    from scripts.agent_ops.v543_s0_probe import (
        load_campaign_config,
        resolve_checkpoints,
    )
    from scripts.campaigns.r3_v53mp_run import _tier_pool_curve_ids
    from pes2ts_core.pesn.models.v543 import CoarseEncoder
    from pes2ts_core.pesn.models.v543.assembly import assemble_v543
    from pes2ts_core.pesn.offline.data.v543_feature_cache import CacheIdentity

    campaign = load_campaign_config(repo_root / campaign_path)
    checkpoints = resolve_checkpoints(campaign, [int(seed)])
    data_cfg, _assets = load_pinned_assets(repo_root / data_path)
    dataset = _open_frozen_dataset(data_cfg)
    labels_v52, endpoint_labels = dataset.label_tables()
    split_path = str((data_cfg.get("split_contract") or {}).get(
        "split_manifest")
        or "/pes2ts_data/v53mp_50k_data/split_manifest_v53mp.json")
    split_doc = json.loads(Path(split_path).read_text(encoding="utf-8"))
    pool_ids = _tier_pool_curve_ids(split_doc, "50k")

    encoder = CoarseEncoder.load(str(checkpoints[int(seed)]["path"]),
                                 int(seed))
    assembly = assemble_v543(encoder, active_track=str(track))
    model = V543Model(assembly)
    source: V543DataSource
    if val_batches is None:
        source = RealDataSource(
            dataset, seed=int(seed), pool_curve_ids=pool_ids,
            cache_identity=CacheIdentity(
                encoder_ckpt_sha256=(encoder_ckpt_sha256
                                     or encoder_digest_v543(encoder))))
    else:                                  # pragma: no cover - smoke budget
        class _SmokeSource(RealDataSource):
            def inner_val_batches(self, limit: int):
                return super().inner_val_batches(
                    min(int(limit), int(val_batches) * V543_BATCH_FRAMES))

        source = _SmokeSource(
            dataset, seed=int(seed), pool_curve_ids=pool_ids,
            cache_identity=CacheIdentity(
                encoder_ckpt_sha256=(encoder_ckpt_sha256
                                     or encoder_digest_v543(encoder))))
    config = V543TrainerConfig(
        track=str(track), decoder_id=str(decoder_id), seed=int(seed),
        stop_after=stop_after, use_cache=bool(use_cache))
    if str(track) != TRACK_MOVE and str(track) != "process":
        raise V543TrainerError("V543_TRACK",
                              f"track must be move/process, got {track!r}")
    return V543Trainer(
        model=model, config=config, source=source, labels_v52=labels_v52,
        endpoint_labels=endpoint_labels, out_dir=out_dir)


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m ...v543_train --track move --decoder M0 --stop-after 200``.

    The WP8 smoke gate runs exactly this (plan §8.1 T1 uses 200 updates).
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="V5.4.3 S1 single-track decoder trainer (frozen encoder)")
    parser.add_argument("--track", default=TRACK_MOVE,
                        choices=[TRACK_MOVE, TRACK_PROCESS])
    parser.add_argument("--decoder", default=None,
                        help="decoder id (defaults to the track baseline)")
    parser.add_argument("--seed", type=int, default=V543_FROZEN_SEEDS[0])
    parser.add_argument("--out", default=None,
                        help="output directory (default "
                             "outputs/pesn/v543_dual_decoder_50k/runs/<id>)")
    parser.add_argument("--stop-after", type=int, default=None,
                        help="stop after N updates (smoke gate); the 37,900 "
                             "dose stays in the config and the run is marked "
                             "non-formal")
    parser.add_argument("--no-cache", action="store_true",
                        help="disable the §7.3 packet cache")
    parser.add_argument("--resume", default=None,
                        help="resume from a matching checkpoint_last.pt")
    args = parser.parse_args(argv)

    from pes2ts_core.pesn.models.v543.registry import baseline_decoder_id

    decoder_id = args.decoder or baseline_decoder_id(args.track)
    out_dir = args.out or str(
        Path("outputs/pesn/v543_dual_decoder_50k/runs")
        / f"V543_{decoder_id}_S{int(args.seed)}")
    try:
        trainer = build_real_trainer(
            track=args.track, decoder_id=decoder_id, seed=int(args.seed),
            out_dir=out_dir, stop_after=args.stop_after,
            use_cache=not args.no_cache)
        if args.resume:
            trainer.resume_from(args.resume)
        metrics = trainer.run()
    except V543TrainerError as error:
        print(f"[v543train] REFUSED {error.code}: {error.detail}",
              file=sys.stderr)
        return 2
    print(json.dumps({key: metrics[key] for key in
                      ("track", "decoder_id", "seed", "updates", "outcome",
                       "is_formal_schedule", "best_val_loss",
                       "encoder_forward_count")},
                     indent=2, sort_keys=True, default=str))
    return 0


__all__ = [
    "LOSS_COEFFICIENTS_V543",
    "MOVE_FRAME_MASK_RULE",
    "PROCESS_FRAME_MASK_RULE",
    "RESUME_IDENTITY_FIELDS_V543",
    "RESUME_IDENTITY_GROUPS_V543",
    "RUN_OUTCOMES_V543",
    "RealDataSource",
    "SKIPPED_NO_GRAD",
    "TrainBatch",
    "V543_BATCH_CURVES",
    "V543_BATCH_FRAMES",
    "V543_BATCH_FRAMES_PER_CURVE",
    "V543_CHECKPOINT_EVERY",
    "V543_DataSource",
    "V543_FROZEN_SEEDS",
    "V543_GRAD_CLIP",
    "V543_INNER_VAL_MAX_SAMPLES",
    "V543_LAMBDA_COS",
    "V543_LAMBDA_DEAD",
    "V543_LAMBDA_ENDPOINT",
    "V543_LAMBDA_PAIR",
    "V543_LAMBDA_Q",
    "V543_LAMBDA_RANK",
    "V543_LAMBDA_STRONG",
    "V543_LR",
    "V543_RANK_MARGIN",
    "V543_SAMPLER_ID",
    "V543_SAMPLER_SPEC_SHA256_PINNED",
    "V543_TOTAL_UPDATES",
    "V543_TRAIN_SCHEMA",
    "V543_VAL_EVERY",
    "V543_WARMUP_UPDATES",
    "V543_WEIGHT_DECAY",
    "V543DataSource",
    "V543Trainer",
    "V543TrainerConfig",
    "V543TrainerError",
    "assert_frozen_loss_coefficients",
    "assert_resume_identity_v543",
    "build_real_trainer",
    "code_fingerprint_v543",
    "encoder_digest_v543",
    "identity_block_v543",
    "lr_at_update",
    "main",
    "move_frame_mask_v543",
    "move_track_loss",
    "process_frame_mask_v543",
    "process_track_loss",
    "track_has_supervision",
    "train_run_v543",
]

