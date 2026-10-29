"""V5.4.1 WP4b — unified MOVE+PROCESS trainer (route E, plan §7).

New loop, ZERO modifications to any legacy file.  Everything shared is
imported and CALLED — never monkeypatched:

* **loss** — ``_loss_on_batch`` / ``key_atom_loss_v53mp`` consumed
  read-only with the INTERNAL arm-C λ set (``λr=0.5, λq=0.5,
  λe=0.012385049119773344``); externally the run only ever reports
  ``objective="unified_move_process"`` (§3.1.4: 新配置/日志/报表/run_id
  不暴露三臂矩阵).
* **features** — ``_FeatureLRU`` / ``_batch_from_positions`` /
  ``build_sample_features_v3`` (level ``F2``).
* **inner-val / τ / evaluation** — ``_inner_val_loss``,
  ``_calibrate_tau_on``, ``evaluate_v53mp`` all run on the FIXED
  inner-val slice only (cap 8192 by default).  ``dev_positions`` is used
  for EXCLUDING dev curves from the pool and nothing else: no dev
  feature, no dev label, no dev eval (§7.2 / WP1 seal).
* **sampler** — the NEW ``v541_strata`` machinery: ``StrataIndex`` built
  once over the inner-train pool, ``curve_batch_positions_v541`` (frozen
  8 curves × 4 frames, four strata, ``(seed, update, salt)`` pure
  protocol) and ``BatchScheduleHasher`` for kill/resume continuity.
* **monitoring** — ``v541_monitor`` wraps the legacy
  ``_grad_monitor_row`` and adds the §7.5 branch / null+reason rows.

Parameter groups are registered EXPLICITLY by name (§7.1: 不再用「非
backbone 全是 head」推断):

==========================  ==========  ==========  ================
group                       W0 (1–400)  W1 (401–…)  lr
==========================  ==========  ==========  ================
``backbone.*``              frozen      joint       ``lr_encoder``
``process_head.*``          train       joint       ``lr_head``
``embedding.*``/``w_out.*``/``alpha``   train   joint   ``lr_encoder``
==========================  ==========  ==========  ================

E0 has no branch group and its ``{param -> lr}`` map is identical to the
legacy arm-C optimizer.  The backbone is frozen with
``requires_grad=False`` only — the forward is NEVER wrapped in
``no_grad()`` (that would cut the new branch's gradient path, §7.5).

Checkpoint/resume (§7.3) carries model/optimizer/update/best/config/
provenance/tau, every RNG state, the strata pool + spec SHAs, the
``BatchScheduleHasher`` state, data/shard position, the
common/embedding/feature/config SHAs, git + environment and a content
digest — written atomically, resumed ONLY from a ``checkpoint_last``
whose identity matches (typed refusal otherwise), with an APPEND-only
``manifest.jsonl``.

Scope: trainer only.  No campaign runner, no ledger writes, no real 50K
run, no dev/confirm access.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from pes2ts_core.pesn.models.key_atom_egnn_v3 import StarEGNNV3
from pes2ts_core.pesn.models.v541 import EMBEDDING_ID_E0, V541Model
from pes2ts_core.pesn.offline.training.feature_prefetch_v53mp import (
    maybe_create_prefetcher,
)
from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import (
    set_seed,
    sha256_file,
)
from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
    CONFIRM_FOLDS_V53MP,
    FEATURE_LEVEL_V53MP,
    PROVENANCE_PLACEHOLDER_KEYS,
    V53MPTrainerConfig,
    V53MPTrainerError,
    _CacheView,
    _FeatureLRU,
    _assert_label_coverage,
    _assert_no_confirm,
    _batch_from_positions,
    _calibrate_tau_on,
    _curve_id_set_of,
    _gpu_memory_metrics,
    _inner_val_loss,
    _loss_on_batch,
    _reset_peak_memory_stats,
    _save as _atomic_save,
    _set_encoder_trainable,
    _set_process_heads_trainable,
    _startup_of,
    _train_q_mean,
    _update_rng,
    _wait_for_eval_vram_headroom,
    apply_cuda_memory_fraction,
    assert_no_legacy_head,
    estimate_endpoint_pos_weight,
    evaluate_v53mp,
    v53mp_pool_split,
)
from pes2ts_core.pesn.offline.training.v541_identity import (
    RUN_OUTCOMES_V541,
    append_manifest,
    content_digest_of,
    env_state,
    git_state,
    positions_sha256,
    rng_state_restore,
    rng_state_snapshot,
    run_outcome_for_exception,
    sha256_json,
)
from pes2ts_core.pesn.offline.training.v541_monitor import (
    PhaseTimer,
    annotated_grad_monitor,
    branch_delta_norm,
    branch_grad_row,
    branch_snapshot,
    embedding_diagnostics,
    group_grad_row,
    host_peak_rss_bytes,
    is_branch_param,
    rms_ratio,
    set_diagnostics_enabled_v541,
)
from pes2ts_core.pesn.offline.training.v541_strata import (
    FRAMES_PER_CURVE_V541,
    SAMPLER_ID_V541,
    BatchScheduleHasher,
    StrataIndex,
    batch_rng_v541,
    curve_batch_positions_v541,
    sampler_spec_sha256,
)

__all__ = [
    "CONFIRM_FOLDS_V53MP",
    "DEGREE_STATS_DIGEST_KEYS",
    "DEGREE_STATS_SCHEMA_V541",
    "RUN_OUTCOMES_V541",
    "V541_FORMAL_TOTAL_UPDATES",
    "V541_TRAIN_SCHEMA",
    "V541TrainerConfig",
    "V541TrainerError",
    "assert_resume_identity",
    "build_checkpoint_v541",
    "config_full_sha256",
    "config_identity_dict",
    "config_identity_sha256",
    "degree_stats_payload_digest",
    "identity_block_v541",
    "is_formal_schedule",
    "load_degree_stats_v541",
    "load_model_v541",
    "make_optimizer_v541",
    "named_param_groups_v541",
    "param_groups_v541",
    "run_outcome_for_exception",
    "set_diagnostics_enabled_v541",
    "set_training_stage_v541",
    "train_run_v541",
]

V541_TRAIN_SCHEMA = "pesn_v541_train_v1"

#: Frozen λ of the unified objective (config/campaigns §training_contract).
V541_LAMBDA_RANK: float = 0.5
V541_LAMBDA_Q: float = 0.5
V541_LAMBDA_ENDPOINT: float = 0.012385049119773344

#: Frozen schedule (§7.1): 400 W0 + 37,500 W1 = 37,900 updates.
V541_W0_UPDATES: int = 400
V541_W1_UPDATES: int = 37500
#: The ONLY total that counts as a formal schedule (§7.4 / §9.1).
V541_FORMAL_TOTAL_UPDATES: int = V541_W0_UPDATES + V541_W1_UPDATES

#: Frozen inner-val cap for selection / τ / final detailed evaluation (§7.2).
V541_INNER_VAL_MAX: int = 8192

#: Log / monitor cadence of the frozen config.
V541_EVERY: int = 100

#: Config fields that are RUN-LENGTH / PROFILE / RESOURCE knobs: they do
#: NOT change a single number of one training update, so they are
#: excluded from the resume-identity ``config_sha256`` (a 20+20 leg must
#: be able to resume into a 40-update run — §7.3 test 1).  Everything
#: that touches numerics (seed, λ, lr, batch, device, val cadence, …)
#: stays inside the identity.
_V541_IDENTITY_EXCLUDED_FIELDS = frozenset({
    "smoke_updates",
    "joint_updates",
    "profile_only",
    "profile_w0_updates",
    "profile_w1_updates",
    "host_limit_gib",
    "memory_high_gib",
    "swap_max",
    "oom_kill_max",
    "concurrency",
})


class V541TrainerError(V53MPTrainerError):
    """V5.4.1 trainer input/state violation (typed, fail-closed).

    ``fields`` names the offending identity keys on a resume refusal.
    """

    def __init__(self, code: str, detail: str,
                 fields: Optional[Sequence[str]] = None) -> None:
        super().__init__(code, detail)
        self.fields = list(fields or ())


def _require(cond: bool, code: str, detail: str,
             fields: Optional[Sequence[str]] = None) -> None:
    if not cond:
        raise V541TrainerError(code, detail, fields=fields)


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------


@dataclass
class V541TrainerConfig(V53MPTrainerConfig):
    """One planned V5.4.1 unified run (§7 / frozen campaign config).

    Subclasses :class:`V53MPTrainerConfig` so every shared loss/eval knob
    keeps ONE definition, then adds the V541-only identity, evaluation
    scope and resource/profile surface.  ``arm`` stays an INTERNAL
    loss-shaping field (always ``"C"`` — the three λ must all be > 0); it
    is never surfaced to logs/metrics/reports, which carry
    ``objective="unified_move_process"`` instead.
    """

    # -- internal loss arm (C only) + external objective -------------------
    arm: str = "C"
    seed: int = 17
    objective: str = "unified_move_process"
    # -- frozen λ of the unified objective ---------------------------------
    lambda_rank: float = V541_LAMBDA_RANK
    lambda_q: float = V541_LAMBDA_Q
    lambda_endpoint: float = V541_LAMBDA_ENDPOINT
    # -- frozen schedule ----------------------------------------------------
    freeze_encoder_updates: int = V541_W0_UPDATES
    joint_updates: int = V541_W1_UPDATES
    # -- frozen evaluation cadence -----------------------------------------
    val_every: int = V541_EVERY
    grad_monitor_every: int = V541_EVERY
    #: the stratified sampler ALWAYS emits 4 frames per curve (§7.2).
    frames_per_curve: int = FRAMES_PER_CURVE_V541
    #: fixed inner-val cap for selection / τ / final detailed evaluation.
    inner_val_max_samples: Optional[int] = V541_INNER_VAL_MAX
    # -- V541 identity ------------------------------------------------------
    embedding_id: str = EMBEDDING_ID_E0
    #: independent branch-init seed (§5.4): REQUIRED for a candidate and
    #: never equal to ``seed`` — the legacy RNG stream stays untouched.
    init_seed: Optional[int] = None
    #: only ``inner_val_only`` exists in this work package.
    eval_scope: str = "inner_val_only"
    common_protocol_sha256: Optional[str] = None
    embedding_spec_sha256: Optional[str] = None
    feature_recipe_sha256: Optional[str] = None
    #: Code identity: the SOURCE-TREE FINGERPRINT (the worktree is dirty, so
    #: the commit alone cannot bind the code) plus the explicit environment
    #: digest.  Both live in the resume identity, so a formal run can only
    #: continue from a checkpoint produced by the SAME code + environment.
    code_sha256: Optional[str] = None
    environment_sha256: Optional[str] = None
    # -- E2 degree statistics (runner-computed, INNER-TRAIN only) ----------
    #: Controlled interface: the runner fits the degree histogram ONCE over a
    #: fixed inner-train index set, caches it, and hands the file + its SHA
    #: here.  ``load_model_v541`` then reads it STRICTLY (file present, SHA
    #: byte-identical, schema/fit-split legal) and pushes it into the branch
    #: via ``set_degree_stats`` exactly once — never on dev/inner-val/confirm
    #: and never recomputed inside ``forward`` (plan §6.6).
    degree_stats_path: Optional[str] = None
    degree_stats_sha256: Optional[str] = None
    # -- resource / profile (recorded; the cgroup side enforces) -----------
    profile_only: bool = False
    profile_w0_updates: Optional[int] = None
    profile_w1_updates: Optional[int] = None
    host_limit_gib: Optional[float] = None
    memory_high_gib: Optional[float] = None
    swap_max: int = 0
    oom_kill_max: int = 0
    concurrency: int = 1
    #: Explicit runner override of the FORMAL-schedule label.  ``None``
    #: (default) derives it: only ``smoke_updates is None`` AND
    #: ``total_updates == 37900`` is the preregistered formal schedule —
    #: a small fixture or the 200-update technical smoke is NEVER
    #: labelled formal (§7.4 / §8.2), while ``run_outcome`` still reports
    #: the honest ``done`` for a completed window.
    formal_run: Optional[bool] = None

    def __post_init__(self) -> None:
        super().__post_init__()               # legacy arm/λ/schedule rules
        _require(self.objective == "unified_move_process", "OBJECTIVE",
                 "objective must be 'unified_move_process' (§3.1.4), got "
                 "{!r}".format(self.objective))
        _require(self.eval_scope == "inner_val_only", "EVAL_SCOPE",
                 "eval_scope must be 'inner_val_only' in WP4b, got "
                 "{!r}".format(self.eval_scope))
        _require(isinstance(self.embedding_id, str)
                 and bool(self.embedding_id), "EMBEDDING_ID",
                 "embedding_id must be a non-empty string")
        if self.embedding_id != EMBEDDING_ID_E0:
            _require(self.init_seed is not None, "INIT_SEED_REQUIRED",
                     "embedding {!r} requires an explicit init_seed "
                     "(independent branch init, plan §5.4)".format(
                         self.embedding_id))
            _require(int(self.init_seed) != int(self.seed),
                     "INIT_SEED_NOT_INDEPENDENT",
                     "init_seed ({}) must differ from the run seed ({}) — "
                     "candidates use an INDEPENDENT init seed".format(
                         self.init_seed, self.seed))
        _require(int(self.frames_per_curve) == FRAMES_PER_CURVE_V541,
                 "BATCH_SHAPE",
                 "frames_per_curve is frozen at 4 (four strata per curve), "
                 "got {}".format(self.frames_per_curve))
        if self.inner_val_max_samples is not None:
            _require(int(self.inner_val_max_samples) >= 1,
                     "INNER_VAL_MAX",
                     "inner_val_max_samples must be >= 1 when set")
        # degree stats are an ATOM: path and SHA travel together or not at all
        _require((self.degree_stats_path is None)
                 == (self.degree_stats_sha256 is None),
                 "DEGREE_STATS_INCOMPLETE",
                 "degree_stats_path and degree_stats_sha256 must be set "
                 "together (got path={!r} sha={!r})".format(
                     self.degree_stats_path, self.degree_stats_sha256))
        if self.degree_stats_sha256 is not None:
            digest = str(self.degree_stats_sha256)
            _require(len(digest) == 64
                     and all(c in "0123456789abcdef" for c in digest),
                     "DEGREE_STATS_SHA",
                     "degree_stats_sha256 must be a 64-hex digest, got "
                     "{!r}".format(self.degree_stats_sha256))
        if self.profile_only:
            _require(
                self.profile_w0_updates is not None
                and int(self.profile_w0_updates) >= 1
                and self.profile_w1_updates is not None
                and int(self.profile_w1_updates) >= 1,
                "PROFILE_WINDOW",
                "profile_only requires profile_w0_updates >= 1 and "
                "profile_w1_updates >= 1 (§8.3 measures W0 and W1 "
                "SEPARATELY — never extrapolated)")
        for name, value in (("host_limit_gib", self.host_limit_gib),
                            ("memory_high_gib", self.memory_high_gib)):
            if value is not None:
                _require(float(value) > 0.0, "RESOURCE_LIMIT",
                         "{} must be > 0 when set, got {}".format(
                             name, value))
        _require(int(self.swap_max) >= 0 and int(self.oom_kill_max) >= 0
                 and int(self.concurrency) >= 1, "RESOURCE_LIMIT",
                 "swap_max/oom_kill_max >= 0 and concurrency >= 1 required")

    # -- schedule aliases (§7.1 names) --------------------------------------
    @property
    def w0_updates(self) -> int:
        """Frozen-core window length (updates 1..400 on a formal run)."""
        return int(self.effective_freeze_updates)

    @property
    def w1_updates(self) -> int:
        """Joint window length (updates 401..37,900 on a formal run)."""
        return int(self.joint_updates)


# ---------------------------------------------------------------------------
# identity SHAs (common / embedding / feature / config)
# ---------------------------------------------------------------------------


def is_formal_schedule(config: V541TrainerConfig) -> bool:
    """Is this config the PREREGISTERED formal schedule? (§7.4 / §9)

    ``True`` only when the run length is the frozen 37,900 updates with
    NO ``smoke_updates`` override — a small fixture or the 200-update
    technical smoke is a real run with a real ``run_outcome``, but it is
    NOT a formal attempt and must never be labelled as one.  A runner may
    pin the label explicitly through ``config.formal_run`` (recorded in
    the identity, so it cannot drift between resume legs).
    """
    explicit = getattr(config, "formal_run", None)
    if explicit is not None:
        return bool(explicit)
    return (config.smoke_updates is None
            and int(config.total_updates) == V541_FORMAL_TOTAL_UPDATES)


def config_identity_dict(config: V541TrainerConfig) -> Dict[str, Any]:
    """Resume-identity projection of the config (§7.3 ``config SHA``).

    Identical to ``asdict(config)`` minus the run-length / profile /
    resource knobs of :data:`_V541_IDENTITY_EXCLUDED_FIELDS` — those
    change WHEN a run stops, never WHAT one update computes.
    """
    blob = asdict(config)
    for key in _V541_IDENTITY_EXCLUDED_FIELDS:
        blob.pop(key, None)
    return blob


def config_identity_sha256(config: V541TrainerConfig) -> str:
    """Canonical SHA of :func:`config_identity_dict`."""
    return sha256_json(config_identity_dict(config))


def config_full_sha256(config: V541TrainerConfig) -> str:
    """Canonical SHA of the WHOLE config (provenance, §12.2)."""
    return sha256_json(asdict(config))


def identity_block_v541(config: V541TrainerConfig,
                        provenance: Mapping[str, Any], *,
                        pool_sha256: Optional[str]) -> Dict[str, Any]:
    """The comparable identity block stored in every checkpoint/metric.

    Every key here is checked on resume — so it contains ONLY values that
    must match (the full config SHA, which legitimately changes when a
    run is extended, lives on the payload instead).
    """
    return {
        "schema": V541_TRAIN_SCHEMA,
        "objective": config.objective,
        "embedding_id": config.embedding_id,
        "init_seed": config.init_seed,
        "seed": int(config.seed),
        "backbone_init": config.backbone_init,
        "tier": provenance.get("tier"),
        "lambda_rank": float(config.lambda_rank),
        "lambda_q": float(config.lambda_q),
        "lambda_endpoint": float(config.lambda_endpoint),
        "eval_scope": config.eval_scope,
        "common_protocol_sha256": config.common_protocol_sha256,
        "embedding_spec_sha256": config.embedding_spec_sha256,
        "feature_recipe_sha256": config.feature_recipe_sha256,
        # code freeze (§17): source fingerprint + environment digest are part
        # of the resume identity — a dirty-worktree edit or an env change can
        # never continue an existing run.
        "code_sha256": config.code_sha256,
        "environment_sha256": config.environment_sha256,
        # E2 degree stats: the cache file's SHA is part of the resume identity
        # (a re-fit / drifted histogram can never resume an existing run).
        "degree_stats_sha256": config.degree_stats_sha256,
        "config_sha256": config_identity_sha256(config),
        "sampler_id": SAMPLER_ID_V541,
        "sampler_spec_sha256": sampler_spec_sha256(),
        "sampler_pool_sha256": pool_sha256,
    }


def assert_resume_identity(state: Mapping[str, Any],
                           config: V541TrainerConfig,
                           provenance: Mapping[str, Any], *,
                           pool_sha256: Optional[str]) -> Dict[str, Any]:
    """Typed refusal unless ``checkpoint_last`` matches THIS identity.

    Every SHA (common / embedding / feature / config / sampler spec+pool),
    the embedding id, the independent init seed, the run seed, the scale
    tier and all three λ must match — otherwise ``RESUME_IDENTITY_MISMATCH``
    with the offending key names on ``exc.fields`` (§7.3: 正式 run 中断后
    只能从匹配 SHA 的 checkpoint_last 恢复).
    """
    _require(isinstance(state, Mapping), "RESUME_INVALID",
             "checkpoint_last payload is not a mapping")
    _require(state.get("schema") == V541_TRAIN_SCHEMA,
             "RESUME_IDENTITY_MISMATCH",
             "checkpoint schema {!r} != {!r}".format(
                 state.get("schema"), V541_TRAIN_SCHEMA),
             fields=["schema"])
    expected = identity_block_v541(config, provenance,
                                   pool_sha256=pool_sha256)
    stored = state.get("identity")
    _require(isinstance(stored, Mapping), "RESUME_IDENTITY_MISMATCH",
             "checkpoint carries no identity block", fields=["identity"])
    mismatches = [k for k, value in expected.items()
                  if stored.get(k) != value]
    if mismatches:
        detail = "; ".join(
            "{}: checkpoint={!r} run={!r}".format(
                k, stored.get(k), expected[k])
            for k in mismatches)
        raise V541TrainerError(
            "RESUME_IDENTITY_MISMATCH",
            "checkpoint_last belongs to a different run — " + detail,
            fields=mismatches)
    return dict(stored)


# ---------------------------------------------------------------------------
# explicit parameter groups (§7.1)
# ---------------------------------------------------------------------------


def named_param_groups_v541(model: torch.nn.Module,
                             config: V541TrainerConfig
                             ) -> Dict[str, List[tuple]]:
    """Classify EVERY parameter by NAME into the three frozen groups.

    A parameter that fits no prefix is a typed refusal — silently falling
    into "head" (the legacy ``not startswith('backbone.')`` rule) is
    exactly what §7.1 forbids, because a new branch would then train at
    ``lr_head = 3e-4``.
    """
    groups: Dict[str, List[tuple]] = {
        "backbone": [], "process_head": [], "embedding": []}
    for name, param in model.named_parameters():
        if name.startswith("backbone."):
            groups["backbone"].append((name, param))
        elif name.startswith("process_head."):
            groups["process_head"].append((name, param))
        elif is_branch_param(name):
            groups["embedding"].append((name, param))
        else:
            raise V541TrainerError(
                "PARAM_GROUP_UNCLASSIFIED",
                "parameter {!r} matches none of backbone.* / process_head.* "
                "/ embedding.*|w_out.*|alpha — refusing to guess a group "
                "(plan §7.1)".format(name))
    if not groups["backbone"]:
        raise V541TrainerError(
            "PARAM_GROUP_EMPTY", "model exposes no backbone.* parameters")
    if not groups["process_head"]:
        raise V541TrainerError(
            "PARAM_GROUP_EMPTY", "model exposes no process_head.* parameters")
    return groups


def param_groups_v541(model: torch.nn.Module,
                       config: V541TrainerConfig) -> List[Dict[str, Any]]:
    """The three explicit AdamW groups: backbone / process_head / embedding.

    ``backbone.*`` (the 3-layer core + the legacy MOVE readout ``w_v`` /
    ``z_head`` + the ``h_legacy`` producers) and the branch share
    ``lr_encoder``; the PROCESS head keeps ``lr_head``.  E0 simply has an
    empty branch group — its resulting ``{param -> lr}`` map is identical
    to the legacy arm-C optimizer.
    """
    named = named_param_groups_v541(model, config)
    groups: List[Dict[str, Any]] = [
        {"name": "backbone",
         "params": [p for _n, p in named["backbone"]],
         "lr": float(config.lr_encoder)},
        {"name": "process_head",
         "params": [p for _n, p in named["process_head"]],
         "lr": float(config.lr_head)},
    ]
    if named["embedding"]:
        groups.append({"name": "embedding",
                       "params": [p for _n, p in named["embedding"]],
                       "lr": float(config.lr_encoder)})
    return groups


def make_optimizer_v541(model: torch.nn.Module,
                        config: V541TrainerConfig) -> torch.optim.Optimizer:
    """AdamW (wd 1e-4, R3 continuity) over the three explicit groups."""
    return torch.optim.AdamW(
        param_groups_v541(model, config),
        weight_decay=float(config.weight_decay))


def param_group_report_v541(model: torch.nn.Module,
                            config: V541TrainerConfig) -> List[Dict[str, Any]]:
    """JSON-safe group summary for ``metrics.json``."""
    return [{"name": g["name"], "lr": float(g["lr"]),
             "n_params": int(sum(int(p.numel()) for p in g["params"])),
             "n_tensors": len(g["params"])}
            for g in param_groups_v541(model, config)]


def set_training_stage_v541(model: torch.nn.Module, *,
                            backbone_trainable: bool) -> None:
    """W0: backbone frozen (``requires_grad=False``), heads + branch live.

    The forward is NEVER put under ``no_grad()`` — that would sever the
    branch's path to the loss (§7.5: 冻结参数不等于对 core 前向使用
    ``no_grad()``).
    """
    _set_process_heads_trainable(model, True)   # process_head + branch
    _set_encoder_trainable(model, bool(backbone_trainable))


# ---------------------------------------------------------------------------
# model construction (legacy RNG order preserved, plan §5.4)
# ---------------------------------------------------------------------------


#: Keys of the FROZEN degree-stats payload that ``stats_sha256`` binds
#: (mirrors ``v541_degree_stats.compute_degree_stats``'s payload).  The
#: loader recomputes the digest over exactly these keys, so ``mu_log`` /
#: ``stats_sha256`` can never drift apart between the fit and the branch.
DEGREE_STATS_DIGEST_KEYS: Tuple[str, ...] = (
    "schema", "split", "split_sha256", "positions_sha256",
    "graph_recipe_sha256", "seed", "seed_sha256", "mu_log",
    "n_positions", "n_unique_graphs", "weighted_graph_total",
    "weighted_node_total", "weighted_edge_total", "weighted_pair_total",
    "degree_histogram",
)

#: The ONLY schema this interface accepts (config ``degree_stats.schema``).
DEGREE_STATS_SCHEMA_V541: str = "pes2ts_v541_degree_stats_v1"


def degree_stats_payload_digest(doc: Mapping[str, Any]) -> str:
    """Recompute ``stats_sha256`` over :data:`DEGREE_STATS_DIGEST_KEYS`."""
    missing = [k for k in DEGREE_STATS_DIGEST_KEYS if k not in doc]
    _require(not missing, "DEGREE_STATS_SCHEMA",
             "degree stats payload is missing the digest key(s) {}".format(
                 missing))
    return sha256_json({k: doc[k] for k in DEGREE_STATS_DIGEST_KEYS})


def load_degree_stats_v541(path: Any, expected_sha256: str) -> Dict[str, Any]:
    """Strict read of the runner-produced degree-stats cache (§6.6).

    Fail-closed on EVERY axis: the file must exist, its bytes must hash to
    ``expected_sha256`` (the SHA recorded in the run identity), the JSON
    must be the frozen ``pes2ts_v541_degree_stats_v1`` envelope, its
    ``fit_split`` must be the inner-train-only one — a histogram fitted on
    dev / inner-val / confirm is refused before it can reach a branch — and
    the two fields the PNA branch actually consumes (``mu_log`` finite and
    > 0, ``stats_sha256`` a real 64-hex digest) must be present AND
    self-consistent (``stats_sha256`` recomputed over the frozen payload).
    """
    digest = str(expected_sha256 or "")
    _require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest),
             "DEGREE_STATS_SHA",
             "degree_stats_sha256 must be a 64-hex digest, got {!r}".format(
                 expected_sha256))
    file_path = Path(str(path))
    _require(file_path.is_file(), "DEGREE_STATS_MISSING",
             "degree stats file not found: {}".format(file_path))
    actual = sha256_file(file_path)
    _require(actual == digest, "DEGREE_STATS_SHA_MISMATCH",
             "{} hashes to {} but the run identity carries {}".format(
                 file_path, actual[:12], digest[:12]),
             fields=["degree_stats_sha256"])
    try:
        doc = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise V541TrainerError(
            "DEGREE_STATS_SCHEMA",
            "{} is not readable JSON: {}".format(file_path, exc)) from exc
    _require(isinstance(doc, Mapping), "DEGREE_STATS_SCHEMA",
             "degree stats payload must be a JSON object, got {}".format(
                 type(doc).__name__))
    _require(doc.get("schema") == DEGREE_STATS_SCHEMA_V541,
             "DEGREE_STATS_SCHEMA",
             "unexpected degree stats schema {!r}".format(doc.get("schema")))
    fit_split = doc.get("fit_split")
    _require(fit_split == "inner-train_only", "DEGREE_STATS_FIT",
             "degree stats must be fitted on inner-train only, got {!r}".format(
                 fit_split))
    forbidden = {str(x) for x in (doc.get("fit_forbidden_splits") or ())}
    _require(str(fit_split) not in forbidden, "DEGREE_STATS_FIT",
             "fit_split {!r} is listed as forbidden".format(fit_split))
    histogram = doc.get("histogram")
    _require(isinstance(histogram, Mapping) and bool(histogram),
             "DEGREE_STATS_SCHEMA",
             "degree stats carry no histogram")
    # --- the two fields set_degree_stats consumes ------------------------
    try:
        mu_log = float(doc.get("mu_log"))
    except (TypeError, ValueError) as exc:
        raise V541TrainerError(
            "DEGREE_STATS_SCHEMA",
            "mu_log must be a real number, got {!r}".format(
                doc.get("mu_log"))) from exc
    _require(math.isfinite(mu_log) and mu_log > 0.0, "DEGREE_STATS_SCHEMA",
             "mu_log must be finite and > 0 (mean of log1p(G_chem degree)), "
             "got {!r}".format(doc.get("mu_log")))
    stats_digest = doc.get("stats_sha256")
    _require(isinstance(stats_digest, str) and len(stats_digest) == 64
             and all(c in "0123456789abcdef" for c in stats_digest),
             "DEGREE_STATS_SCHEMA",
             "stats_sha256 must be a 64-hex digest, got {!r}".format(
                 stats_digest))
    recomputed = degree_stats_payload_digest(doc)
    _require(recomputed == stats_digest, "DEGREE_STATS_SHA_MISMATCH",
             "stats_sha256 {} does not match the recomputed payload digest "
             "{} — the envelope was edited after the fit".format(
                 stats_digest[:12], recomputed[:12]),
             fields=["degree_stats_sha256"])
    return dict(doc)


def _degree_stats_target(model: torch.nn.Module) -> Any:
    """The object that owns ``set_degree_stats`` (model hook or branch)."""
    if callable(getattr(model, "set_degree_stats", None)):
        return model
    embedding = getattr(model, "embedding", None)
    if embedding is not None and callable(
            getattr(embedding, "set_degree_stats", None)):
        return embedding
    return None


def _load_degree_stats_into(model: torch.nn.Module,
                            config: V541TrainerConfig) -> Optional[Dict[str, Any]]:
    """Wire the controlled degree-stats interface (defence in depth).

    The real PNA branch consumes exactly two values —
    ``set_degree_stats(mu_log, stats_sha256)`` — so the envelope is read
    strictly first (see :func:`load_degree_stats_v541`) and then handed over
    in that exact two-argument form, ONCE, at construction time (never
    inside ``forward``).

    * stats configured + a hook present → strict read, then ONE
      ``set_degree_stats(mu_log, stats_sha256)`` call;
    * stats configured + NO hook → typed refusal (the candidate cannot
      silently ignore its frozen degree statistics);
    * no stats + a hook present → typed refusal (the candidate would run
      with unfitted degree scalers);
    * E0 / candidates without degree stats + no stats → no-op.
    """
    target = _degree_stats_target(model)
    path = config.degree_stats_path
    digest = config.degree_stats_sha256
    if path is None and digest is None:
        _require(target is None, "DEGREE_STATS_REQUIRED",
                 "embedding {!r} exposes set_degree_stats but the config "
                 "carries no degree_stats_path/degree_stats_sha256".format(
                     config.embedding_id))
        return None
    _require(path is not None and digest is not None, "DEGREE_STATS_INCOMPLETE",
             "degree_stats_path and degree_stats_sha256 must travel together")
    payload = load_degree_stats_v541(path, digest)
    _require(target is not None, "DEGREE_STATS_NOT_SUPPORTED",
             "config carries degree stats for {!r} but the model exposes no "
             "set_degree_stats hook".format(config.embedding_id))
    target.set_degree_stats(payload["mu_log"], payload["stats_sha256"])
    return payload


def load_model_v541(config: V541TrainerConfig, *,
                    r3_ckpt: Any = None,
                    device: str = "cpu") -> V541Model:
    """Build the run model — R3 strict warmstart or scratch, E0 or branch.

    Construction order is the legacy one (``StarEGNNV3`` →
    ``ReactionProcessHeadV53MP`` under the seeded global stream) and the
    branch is then built inside ``V541Model``'s ``fork_rng`` so the legacy
    RNG sequence is untouched.  ``init_seed`` (candidates only) is the
    INDEPENDENT branch seed.

    When the config carries ``degree_stats_path``/``degree_stats_sha256``
    the file is read STRICTLY after construction and pushed into the branch
    through ``set_degree_stats(mu_log, stats_sha256)`` — once, at
    construction time (never per forward).
    """
    if config.backbone_init == "r3":
        _require(r3_ckpt is not None, "R3_CHECKPOINT_MISSING",
                 "backbone_init='r3' requires r3_ckpt")
        payload = torch.load(str(r3_ckpt), map_location="cpu",
                             weights_only=False)
        _require(isinstance(payload, Mapping) and "model" in payload,
                 "R3_CHECKPOINT_FORMAT",
                 "{} is not an R3 trainer checkpoint".format(r3_ckpt))
        backbone = StarEGNNV3()
        backbone.load_state_dict(payload["model"], strict=True)
        model = V541Model(backbone=backbone,
                          embedding_id=config.embedding_id,
                          embedding_init_seed=config.init_seed)
    else:
        model = V541Model(embedding_id=config.embedding_id,
                          embedding_init_seed=config.init_seed)
    assert_no_legacy_head(model)
    set_diagnostics_enabled_v541(model, True)
    _load_degree_stats_into(model, config)
    return model.to(device)


# ---------------------------------------------------------------------------
# checkpoint payload (§7.3)
# ---------------------------------------------------------------------------


def build_checkpoint_v541(*, model: torch.nn.Module,
                          optimizer: torch.optim.Optimizer,
                          update: int, best_val: float, best_update: int,
                          config: V541TrainerConfig,
                          seed_map: Mapping[str, int],
                          provenance: Mapping[str, Any],
                          identity: Mapping[str, Any],
                          schedule: BatchScheduleHasher,
                          data_position: Mapping[str, Any],
                          stats: Mapping[str, Any],
                          monitor_state: Mapping[str, Any],
                          git: Mapping[str, Any],
                          env: Mapping[str, Any],
                          tau_block: Optional[Mapping[str, Any]] = None,
                          ) -> Dict[str, Any]:
    """One atomic checkpoint payload — every §7.3 field, digest last."""
    payload: Dict[str, Any] = {
        "schema": V541_TRAIN_SCHEMA,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": int(update),
        "best_val": float(best_val),
        "best_update": int(best_update),
        "config": asdict(config),
        "config_full_sha256": config_full_sha256(config),
        "identity": dict(identity),
        "provenance": dict(provenance),
        "seeds": dict(seed_map),
        # --- RNG (python / numpy / torch / torch_cuda) --------------------
        "rng": rng_state_snapshot(config.device),
        # --- sampler / schedule -------------------------------------------
        "sampler": {
            "sampler_id": SAMPLER_ID_V541,
            "spec_sha256": sampler_spec_sha256(),
            "pool_sha256": identity.get("sampler_pool_sha256"),
            "batch_curves": int(config.batch_curves),
            "frames_per_curve": int(config.frames_per_curve),
        },
        "schedule": schedule.state_dict(),
        "sampler_counters": dict(stats.get("fallback", {})),
        "monitor_state": dict(monitor_state),
        # --- data / shard position ----------------------------------------
        "data_position": dict(data_position),
        # --- git / env -----------------------------------------------------
        "git": dict(git),
        "env": dict(env),
        # --- τ (filled after calibration; None until then) -----------------
        "tau_end_calibration": (dict(tau_block) if tau_block else None),
    }
    payload["content_digest_sha256"] = content_digest_of(payload)
    return payload


def _save_checkpoint(path: Path, payload: Mapping[str, Any],
                     manifest: Path, event: str) -> None:
    """Atomic save + APPEND-only manifest record (§7.3: 追加而非覆盖)."""
    _atomic_save(path, dict(payload))
    append_manifest(manifest, {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "path": str(path),
        "update": int(payload.get("update", -1)),
        "file_sha256": sha256_file(path),
        "content_digest_sha256": payload.get("content_digest_sha256"),
        "config_sha256": (payload.get("identity") or {}).get("config_sha256"),
    })


# ---------------------------------------------------------------------------
# run preparation (shared by the formal loop and the profile window)
# ---------------------------------------------------------------------------


def _positions_block(train_pos: Sequence[int],
                     inner_val_pos: Sequence[int],
                     dev_curves: Sequence[str],
                     provenance: Mapping[str, Any],
                     n_samples: int) -> Dict[str, Any]:
    """Data/shard position of THIS run (§7.3 「数据/shard 位置」)."""
    return {
        "n_samples": int(n_samples),
        "n_train_positions": int(len(train_pos)),
        "n_inner_val_positions": int(len(inner_val_pos)),
        "train_positions_sha256": positions_sha256(train_pos),
        "inner_val_positions_sha256": positions_sha256(inner_val_pos),
        "dev_curve_ids_sha256": sha256_json(sorted(
            str(c) for c in dev_curves)),
        "dev_curve_ids": sorted(str(c) for c in dev_curves),
        "data_manifest_sha256": provenance.get("data_manifest_sha256"),
        "split_manifest_sha256": provenance.get("split_manifest_sha256"),
        "shard": provenance.get("shard_path", provenance.get("shard")),
        "note": ("dev positions were used ONLY to exclude dev curves from "
                 "the pool — no dev feature, label or eval was built"),
    }


def _prepare_run(dataset: Any, labels_v52: Any, endpoint_labels: Any,
                 config: V541TrainerConfig, out_dir: Any, *,
                 r3_ckpt: Any, dev_positions: Optional[Sequence[int]],
                 train_positions: Optional[Sequence[int]],
                 provenance: Optional[Mapping[str, Any]],
                 feature_prefetcher: Any,
                 feature_workers: int,
                 resume: bool) -> SimpleNamespace:
    """Position sets, sampler index, model, optimizer, identity, resume."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    prov: Dict[str, Any] = {k: None for k in PROVENANCE_PLACEHOLDER_KEYS}
    prov.update(dict(provenance or {}))
    seed_map = set_seed(config.seed)
    device = config.device
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        raise V541TrainerError("CUDA_UNAVAILABLE", "device=cuda but no CUDA")
    apply_cuda_memory_fraction(config)
    _reset_peak_memory_stats(device)

    t_startup = time.time()
    timers = PhaseTimer()

    # --- position sets (fail-closed BEFORE any model/loss work) -----------
    samples = dataset.samples
    n_samples = len(samples)
    _require(dev_positions is not None and len(list(dev_positions)) > 0,
             "DEV_POSITIONS",
             "dev_positions must be non-empty — they exist to EXCLUDE the "
             "dev curves from the pool (never to be evaluated here)")
    dev_pos = [int(p) for p in dev_positions]
    for p in dev_pos:
        _require(0 <= p < n_samples, "POSITION_RANGE",
                 "position {} outside 0..{}".format(p, n_samples - 1))
    dev_curve_set = _curve_id_set_of(samples, dev_pos)

    if train_positions is None:
        st = _startup_of(samples)
        if st is not None:
            dev_rows = np.asarray(
                sorted(st.curve_row[c] for c in dev_curve_set
                       if c in st.curve_row), dtype=np.int64)
            pool = np.flatnonzero(
                ~np.isin(np.asarray(st.curve_idx), dev_rows)
            ).astype(np.int64).tolist()
        else:
            pool = [i for i in range(n_samples)
                    if str(samples[i].curve_id) not in dev_curve_set]
    else:
        pool = [int(p) for p in train_positions]
    for p in pool:
        _require(0 <= p < n_samples, "POSITION_RANGE",
                 "position {} outside 0..{}".format(p, n_samples - 1))
    pool_curves = _curve_id_set_of(samples, pool)
    train_pos, inner_val_pos = v53mp_pool_split(
        samples, pool_curve_ids=sorted(pool_curves), seed=config.seed,
        inner_val_fraction=config.inner_val_fraction)
    if train_positions is not None and \
            set(pool) != set(train_pos) | set(inner_val_pos):
        raise V541TrainerError(
            "TRAIN_POOL_NOT_CURVE_COMPLETE",
            "train_positions must be the FULL pool (train + inner-val "
            "curves) returned by v53mp_pool_split; got {} positions, the "
            "curve-level split covers {}".format(
                len(pool), len(set(train_pos) | set(inner_val_pos))))
    _assert_no_confirm(samples, train_pos, "train")
    _assert_no_confirm(samples, inner_val_pos, "inner-val")
    _assert_no_confirm(samples, dev_pos, "dev")

    # --- fixed inner-val cap (selection + τ + final detailed evaluation) --
    if config.inner_val_max_samples is not None and \
            len(inner_val_pos) > int(config.inner_val_max_samples):
        rng_iv = np.random.RandomState(int(config.seed) ^ 0x1E7A1)
        inner_val_pos = sorted(rng_iv.choice(
            list(inner_val_pos), size=int(config.inner_val_max_samples),
            replace=False).tolist())

    # --- NEW stratified sampler index (built ONCE over inner-train) -------
    index = StrataIndex.build(samples, train_pos, dataset=dataset)
    train_curves = _curve_id_set_of(samples, train_pos)
    val_curves = _curve_id_set_of(samples, inner_val_pos)
    dev_curves = _curve_id_set_of(samples, dev_pos)
    _require(not (dev_curves & (train_curves | val_curves)),
             "DEV_TRAIN_OVERLAP",
             "dev curves overlap the train pool: {}".format(
                 sorted(dev_curves & (train_curves | val_curves))))
    _require(bool(index.curves), "SAMPLER_POOL_EMPTY",
             "StrataIndex covers no inner-train curve")

    # --- label coverage over train + inner-val ONLY (dev never touched) ---
    _assert_label_coverage(samples, labels_v52, endpoint_labels,
                           [train_pos, inner_val_pos], n_samples)
    ep_weights = estimate_endpoint_pos_weight(
        samples, endpoint_labels, train_pos, seed=config.seed)
    endpoint_pos_weight = float(ep_weights["pos_weight"])

    # --- model (seeded AFTER the deterministic setup, V5.2 discipline) ----
    set_seed(config.seed)
    model = load_model_v541(config, r3_ckpt=r3_ckpt, device=device)
    set_training_stage_v541(
        model, backbone_trainable=(1 > config.effective_freeze_updates))
    optimizer = make_optimizer_v541(model, config)

    identity = identity_block_v541(config, prov,
                                   pool_sha256=index.pool_sha256)
    git = git_state()
    env = env_state()

    # --- resume (identity-checked; profile windows never resume) ----------
    last_path = out / "checkpoint_last.pt"
    best_path = out / "checkpoint_best.pt"
    manifest_path = out / "manifest.jsonl"
    start_update = 1
    best_val = float("inf")
    best_update = 0
    schedule = BatchScheduleHasher()
    stats: Dict[str, Any] = {"cache_hits": 0, "cache_misses": 0,
                             "fallback": {}}
    monitor_state: Dict[str, Any] = {"branch_prev": None,
                                     "monitor_count": 0}
    if resume and last_path.is_file():
        state = torch.load(str(last_path), map_location="cpu",
                           weights_only=False)
        assert_resume_identity(state, config, prov,
                               pool_sha256=index.pool_sha256)
        # §7.3: never load a payload that was edited after it was written.
        # The digest is recomputed over EVERYTHING except the digest key
        # itself, so a stale or missing digest types-refuses before a
        # single tensor reaches the model/optimizer.
        stored_digest = state.get("content_digest_sha256")
        actual_digest = content_digest_of(state)
        if not isinstance(stored_digest, str) or actual_digest != stored_digest:
            raise V541TrainerError(
                "RESUME_CONTENT_DIGEST",
                "checkpoint_last content digest mismatch — payload was "
                "modified after it was written (stored={!r}, recomputed={!r}); "
                "refusing to load".format(stored_digest, actual_digest),
                fields=["content_digest_sha256"])
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_update = int(state["update"]) + 1
        best_val = float(state["best_val"])
        best_update = int(state["best_update"])
        schedule = BatchScheduleHasher.from_state(state["schedule"])
        rng_state_restore(state["rng"])
        stats["fallback"] = dict(state.get("sampler_counters") or {})
        monitor_state = dict(state.get("monitor_state") or
                             {"branch_prev": None, "monitor_count": 0})
        append_manifest(manifest_path, {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "resume",
            "from_update": int(state["update"]),
            "next_update": int(start_update),
            "content_digest_sha256": state.get("content_digest_sha256"),
            "config_sha256": identity.get("config_sha256"),
        })
    set_training_stage_v541(
        model,
        backbone_trainable=(start_update > config.effective_freeze_updates))

    lru = _FeatureLRU(dataset, capacity=int(config.feature_cache_capacity),
                      level=FEATURE_LEVEL_V53MP, intent_off=False)
    val_lru = _FeatureLRU(dataset,
                          capacity=max(len(inner_val_pos), 1) + 64,
                          level=FEATURE_LEVEL_V53MP, intent_off=False)

    timers.add("startup", "wall_s", time.time() - t_startup)
    return SimpleNamespace(
        out=out, config=config, prov=prov, seed_map=seed_map, device=device,
        dataset=dataset, labels=labels_v52, endpoint=endpoint_labels,
        samples=samples, n_samples=n_samples,
        train_pos=train_pos, inner_val_pos=inner_val_pos, dev_pos=dev_pos,
        dev_curves=dev_curves, train_curves=train_curves,
        val_curves=val_curves, pool_curves=pool_curves,
        index=index, endpoint_pos_weight=endpoint_pos_weight,
        ep_weights=ep_weights, lru=lru, val_lru=val_lru,
        model=model, optimizer=optimizer, schedule=schedule,
        identity=identity, git=git, env=env, timers=timers, stats=stats,
        monitor_state=monitor_state,
        prefetcher=feature_prefetcher, feature_workers=int(feature_workers),
        log_path=out / "train_log.jsonl", last_path=last_path,
        best_path=best_path, manifest_path=manifest_path,
        start_update=start_update, best_val=best_val,
        best_update=best_update, r3_ckpt=r3_ckpt,
        eval_gate_waits=[], t_run_start=time.time())


# ---------------------------------------------------------------------------
# per-update helpers
# ---------------------------------------------------------------------------


def _log_row(path: Path, row: Mapping[str, Any]) -> None:
    _log_rows(path, [row])


def _log_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Append JSON lines in ONE open (the per-update diag flags are
    buffered so a 37,900-update run still pays one append per monitor
    step, not one per update)."""
    if not rows:
        return
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True,
                                    default=str) + "\n")


def _move_rows(records: Sequence[Any]) -> int:
    """MOVE-eligible rows of this batch (``frame_mask`` true count)."""
    return sum(1 for rec in records
               if str(getattr(rec, "label_status", "")) == "ok")


def _embedding_flag(model: torch.nn.Module) -> Optional[bool]:
    """The branch's own diagnostics toggle, or ``None`` when it has none."""
    emb = getattr(model, "embedding", None)
    if emb is None or not hasattr(emb, "diagnostics_enabled"):
        return None
    return bool(getattr(emb, "diagnostics_enabled"))


#: Key aliases per quantity the §7.5 log must carry.  The primary name is
#: first; the variants are the names the WP3 providers emit today.  A
#: quantity with NO key in either source stays ``None`` + reason — never
#: a fabricated 0.
_QUANTITY_KEYS_V541: Dict[str, Tuple[str, ...]] = {
    "n_edges": ("n_edges", "edge_count"),
    "n_triplets": ("n_triplets", "n_angle_triplets", "triplet_count"),
    "zero_neighbor_frac": ("zero_neighbor_frac", "zero_neighbors_frac",
                           "zero_neighbor_fraction"),
    "zero_angle_frac": ("zero_angle_frac", "zero_angles_frac",
                        "zero_angle_fraction"),
}

#: Why a branch-side quantity is missing, keyed by ``embedding_diagnostics``
#: status (see v541_monitor.embedding_diagnostics).  ``empty`` is exactly
#: the WP3 ``diagnostics=False`` contract (counts={}).
_EMB_ABSENT_REASON_V541: Dict[str, str] = {
    "absent_no_branch": "e0_no_branch",
    "attr_absent": "embedding_diagnostics_unavailable",
    "attr_not_a_mapping": "embedding_diagnostics_unavailable",
    "empty": "embedding_diagnostics_empty",
    "ok": "key_not_reported_by_embedding",
}


def _pick_diag(sources: Sequence[Mapping[str, Any]],
               keys: Sequence[str]) -> Optional[Any]:
    """First non-``None`` value of any alias across the sources, in order."""
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in keys:
            value = source.get(key)
            if value is not None:
                return value
    return None


def _edge_triplet_block(diag: Mapping[str, Any],
                        emb_diag: Mapping[str, Any],
                        emb_status: str) -> Dict[str, Any]:
    """edge / triplet / zero-neighbor / zero-angle accounting (§7.5).

    Two sources, merged WITHOUT touching the WP3 files: the model-level
    ``V541Model.diagnostics`` first (``n_nodes``/``n_edges`` live there),
    then the branch's own ``last_diagnostics``/``diagnostics`` for the
    provider counts.  Every missing quantity carries an explicit reason
    (``e0_no_branch`` / ``embedding_diagnostics_unavailable`` /
    ``embedding_diagnostics_empty`` for a ``diagnostics=False`` build /
    ``key_not_reported_by_embedding``) — nothing is ever reported as 0
    when it was not measured.
    """
    reasons: Dict[str, str] = {}
    n_nodes = _pick_diag((diag, emb_diag), ("n_nodes",))
    quantities: Dict[str, Optional[Any]] = {}
    for name, keys in _QUANTITY_KEYS_V541.items():
        value = _pick_diag((diag, emb_diag), keys)
        quantities[name] = value
        if value is None:
            if name == "n_edges" and not diag:
                reasons[name] = "model_diagnostics_disabled_or_e0"
            else:
                reasons[name] = _EMB_ABSENT_REASON_V541.get(
                    emb_status, "not_reported")
    return {
        "n_nodes": n_nodes,
        "n_edges": quantities["n_edges"],
        "n_triplets": quantities["n_triplets"],
        "zero_neighbor_frac": quantities["zero_neighbor_frac"],
        "zero_angle_frac": quantities["zero_angle_frac"],
        "reasons": reasons,
        "embedding_diagnostics_status": emb_status,
        "source": ("V541Model.diagnostics + model.embedding."
                   "last_diagnostics|diagnostics (WP3 provider counts)"),
    }


def _batch_positions_v541(st: SimpleNamespace, update: int) -> List[int]:
    """One update's stratified batch — the frozen ``(seed, update, 0)`` draw."""
    return curve_batch_positions_v541(
        st.index,
        rng=batch_rng_v541(int(st.config.seed), int(update)),
        batch_curves=int(st.config.batch_curves),
        counters=st.stats["fallback"])


def _data_position(st: SimpleNamespace) -> Dict[str, Any]:
    return _positions_block(st.train_pos, st.inner_val_pos, st.dev_curves,
                            st.prov, st.n_samples)


def _new_payload(st: SimpleNamespace, update: int) -> Dict[str, Any]:
    return build_checkpoint_v541(
        model=st.model, optimizer=st.optimizer, update=update,
        best_val=st.best_val, best_update=st.best_update,
        config=st.config, seed_map=st.seed_map, provenance=st.prov,
        identity=st.identity, schedule=st.schedule,
        data_position=_data_position(st), stats=st.stats,
        monitor_state=st.monitor_state, git=st.git, env=st.env)


def _record_update(st: SimpleNamespace, update: int, phase: str, *,
                   loss_out: Mapping[str, Any], positions: Sequence[int],
                   monitor: bool, diag: Mapping[str, float],
                   emb_diag: Mapping[str, Any], emb_status: str,
                   grad_row: Optional[Mapping[str, Any]],
                   group_row: Optional[Mapping[str, Any]],
                   branch_row: Optional[Mapping[str, Any]],
                   alpha_before: Optional[float],
                   alpha_after: Optional[float],
                   delta_norm: Optional[float],
                   stage_switch: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Build one rich JSONL row of ``train_log.jsonl`` (§7.1 / §7.5)."""
    cfg = st.config
    row: Dict[str, Any] = {
        "event": "train",
        "update": int(update),
        "phase": phase,
        "objective": cfg.objective,
        "embedding_id": cfg.embedding_id,
        "loss": float(loss_out["loss"].detach().cpu()),
        "l_move": float(loss_out["l_move"].detach().cpu()),
        "l_rank": float(loss_out["l_rank"].detach().cpu()),
        "l_q": float(loss_out["l_q"].detach().cpu()),
        "l_endpoint": float(loss_out["l_endpoint"].detach().cpu()),
        "n_rank_pairs": int(loss_out["n_rank_pairs"]),
        "n_q_rows": int(loss_out["n_q_rows"]),
        "n_endpoint_rows": int(loss_out["n_endpoint_rows"]),
        "batch_positions": [int(p) for p in positions],
        "fallback_counts": dict(st.stats["fallback"]),
        "diagnostics_enabled": bool(
            getattr(st.model, "diagnostics_enabled", False)),
        "embedding_diagnostics_enabled": _embedding_flag(st.model),
        "encoder_trainable": any(
            p.requires_grad for p in st.model.backbone.parameters()),
        "elapsed_s": round(time.time() - st.t_run_start, 3),
        "phase_timing": st.timers.as_dict().get(phase),
        "resources": {
            "host_peak_rss_bytes": host_peak_rss_bytes(),
            **_gpu_memory_metrics(cfg),
        },
    }
    if stage_switch is not None:
        row["stage_switch"] = dict(stage_switch)
    if monitor:
        row["grad_monitor"] = dict(grad_row or {})
        row["param_group_grads"] = dict(group_row or {})
        row["branch"] = dict(branch_row or {})
        row["alpha_before_step"] = alpha_before
        row["alpha_after_step"] = alpha_after
        row["branch_param_delta_norm"] = delta_norm
        row["rms_h_legacy"] = diag.get("rms_h_legacy")
        row["rms_alpha_W_f"] = diag.get("rms_alpha_W_f")
        row["rms_ratio"] = rms_ratio(diag.get("rms_alpha_W_f"),
                                      diag.get("rms_h_legacy"))
        row["diagnostics"] = dict(diag)
        row["embedding_diagnostics"] = dict(emb_diag)
        row["embedding_diagnostics_status"] = emb_status
        row["edge_triplet_zero_neighbor"] = _edge_triplet_block(
            diag, emb_diag, emb_status)
    else:
        reason = {"skipped": True, "reason": "not_monitor_step"}
        row["grad_monitor"] = dict(reason)
        row["param_group_grads"] = dict(reason)
        row["branch"] = dict(reason)
    return row


def _stage_switch_row(st: SimpleNamespace, update: int,
                      phase: str) -> Dict[str, Any]:
    """W0 → W1 evidence: core is IN the optimizer and grads reach it (§7.5)."""
    backbone = list(st.model.backbone.parameters())
    grads = [p.grad for p in backbone if p.grad is not None]
    max_abs = (max(float(g.abs().max()) for g in grads)
               if grads else None)
    return {
        "event": "stage_switch",
        "update": int(update),
        "phase": phase,
        "core_requires_grad": all(p.requires_grad for p in backbone),
        "n_core_params": len(backbone),
        "n_core_params_with_grad": len(grads),
        "core_grad_max_abs": max_abs,
        "core_reachable": bool(grads) and bool(max_abs is not None
                                               and max_abs > 0.0),
        "note": ("frozen != no_grad: the forward ran WITH graph so the "
                 "branch stayed differentiable (plan §7.5)"),
    }


# ---------------------------------------------------------------------------
# formal training loop
# ---------------------------------------------------------------------------


def _run_formal(st: SimpleNamespace) -> Dict[str, Any]:
    cfg = st.config
    total = int(cfg.total_updates)
    monitor_every = max(int(cfg.grad_monitor_every), 1)
    t_loop = time.time()
    last_loss: Optional[float] = None
    updates_run = 0
    pending_flags: List[Dict[str, Any]] = []

    for update in range(st.start_update, total + 1):
        t_iter = time.time()
        phase = "w0" if update <= cfg.effective_freeze_updates else "w1"
        stage_switch: Optional[Dict[str, Any]] = None
        stage_switch_pending = (update == cfg.effective_freeze_updates + 1)
        if stage_switch_pending:
            set_training_stage_v541(st.model, backbone_trainable=True)

        monitor = (update % monitor_every == 0 or update == 1
                   or update == total)
        log_step = monitor or (update % V541_EVERY == 0)

        # --- sampler draw (pure (seed, update, 0)) + schedule chain --------
        positions = _batch_positions_v541(st, update)
        st.schedule.record(update, positions)

        # --- diagnostics only on monitor steps (candidate-only sync cost) --
        set_diagnostics_enabled_v541(st.model, monitor)

        # --- data lane ------------------------------------------------------
        t_data = time.time()
        for p in positions:
            if st.lru.contains(int(p)):
                st.stats["cache_hits"] += 1
            else:
                st.stats["cache_misses"] += 1
        batch, records = _batch_from_positions(
            st.dataset, st.lru, positions, st.device,
            prefetcher=st.prefetcher)
        data_s = time.time() - t_data

        # --- core lane: forward, monitor, backward, step --------------------
        t_core = time.time()
        st.optimizer.zero_grad(set_to_none=True)
        loss_out = _loss_on_batch(
            st.model, batch, records, st.labels, st.endpoint, cfg,
            endpoint_pos_weight=st.endpoint_pos_weight,
            pair_rng=_update_rng(int(cfg.seed), int(update), salt=1))
        diag: Dict[str, float] = dict(
            getattr(st.model, "diagnostics", {}) or {})
        emb_diag, emb_status = embedding_diagnostics(st.model)
        alpha_before = (float(st.model.alpha.detach())
                        if getattr(st.model, "alpha", None) is not None
                        else None)
        grad_row = group_row = branch_row = None
        if monitor:
            grad_row = annotated_grad_monitor(
                loss_out, list(st.model.backbone.parameters()), cfg,
                move_rows=_move_rows(records))
            group_row = group_grad_row(st.model, loss_out["loss"])
            branch_row = branch_grad_row(st.model, loss_out["loss"])
        prev_snap = st.monitor_state.get("branch_prev")
        if bool(loss_out["loss"].requires_grad):
            loss_out["loss"].backward()
            trainable = [p for p in st.model.parameters() if p.requires_grad]
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable,
                                               float(cfg.grad_clip))
            st.optimizer.step()
        else:
            _log_row(st.log_path, {
                "event": "skipped_no_grad",
                "update": int(update),
                "reason": ("loss has no grad path (no active supervised "
                           "rows in this batch)")})
        core_s = time.time() - t_core

        alpha_after = (float(st.model.alpha.detach())
                       if getattr(st.model, "alpha", None) is not None
                       else None)
        delta_norm: Optional[float] = None
        if monitor:
            delta_norm = branch_delta_norm(st.model, prev_snap)
            st.monitor_state["branch_prev"] = branch_snapshot(st.model)
            st.monitor_state["monitor_count"] = int(
                st.monitor_state.get("monitor_count", 0)) + 1

        last_loss = float(loss_out["loss"].detach().cpu())
        st.timers.add(phase, "data_s", data_s)
        st.timers.add(phase, "core_s", core_s)
        st.timers.add(phase, "wall_s", time.time() - t_iter)
        st.timers.count(phase)
        updates_run += 1

        if stage_switch_pending:
            stage_switch = _stage_switch_row(st, update, phase)
        if not monitor:
            # cheap per-update evidence that the candidate-only scalar
            # diagnostics are OFF outside monitor steps (§7.5 cost rule);
            # buffered so the formal run still appends once per monitor.
            pending_flags.append({
                "event": "diag_flag",
                "update": int(update),
                "phase": phase,
                "diagnostics_enabled": False,
                "embedding_diagnostics_enabled": _embedding_flag(st.model),
            })
        if stage_switch_pending and not log_step:
            _log_rows(st.log_path, pending_flags + [stage_switch])
            pending_flags = []
        if log_step:
            row = _record_update(
                st, update, phase, loss_out=loss_out, positions=positions,
                monitor=monitor, diag=diag, emb_diag=emb_diag,
                emb_status=emb_status, grad_row=grad_row,
                group_row=group_row, branch_row=branch_row,
                alpha_before=alpha_before,
                alpha_after=alpha_after, delta_norm=delta_norm,
                stage_switch=stage_switch)
            _log_rows(st.log_path, pending_flags + [row])
            pending_flags = []

        # --- inner-val selection sweep (selection + τ stay inner-val) ------
        if update % max(int(cfg.val_every), 1) == 0 or update == total:
            _wait_for_eval_vram_headroom(cfg, st.eval_gate_waits)
            t_eval = time.time()
            val = _inner_val_loss(
                st.model, st.dataset, st.val_lru, st.inner_val_pos,
                st.labels, st.endpoint, cfg,
                endpoint_pos_weight=st.endpoint_pos_weight, update=update,
                prefetcher=st.prefetcher)
            st.timers.add(phase, "eval_s", time.time() - t_eval)
            _log_row(st.log_path, {"event": "inner_val", "update": update,
                                   "inner_val_loss": val})
            if val < st.best_val:
                st.best_val = float(val)
                st.best_update = int(update)
                t_ck = time.time()
                _save_checkpoint(st.best_path, _new_payload(st, update),
                                 st.manifest_path, "checkpoint_best")
                st.timers.add(phase, "checkpoint_s", time.time() - t_ck)
            t_ck = time.time()
            _save_checkpoint(st.last_path, _new_payload(st, update),
                             st.manifest_path, "checkpoint_last")
            st.timers.add(phase, "checkpoint_s", time.time() - t_ck)

    if pending_flags:
        _log_rows(st.log_path, pending_flags)
        pending_flags = []

    return {"last_loss": last_loss, "total_updates": total,
            "updates_run": int(updates_run),
            "loop_wall_s": round(time.time() - t_loop, 3)}


# ---------------------------------------------------------------------------
# profile-only window (§8.3: W0 and W1 measured SEPARATELY)
# ---------------------------------------------------------------------------


def _run_profile(st: SimpleNamespace) -> Dict[str, Any]:
    """Short, EXPLICITLY marked profile window — never a formal run.

    Measures frozen-core (W0) and joint (W1) updates apart, writes
    ``profile.json`` + a ``profile_only`` metrics file, and touches
    NEITHER the formal schedule (400 + 37,500) NOR the checkpoint /
    manifest lane of a real run.
    """
    cfg = st.config
    w0 = int(cfg.profile_w0_updates)
    w1 = int(cfg.profile_w1_updates)
    total = w0 + w1
    monitor_every = max(int(cfg.grad_monitor_every), 1)
    set_training_stage_v541(st.model, backbone_trainable=False)
    last_loss: Optional[float] = None
    updates_run = 0

    for update in range(1, total + 1):
        t_iter = time.time()
        phase = "profile_w0" if update <= w0 else "profile_w1"
        if update == w0 + 1:
            set_training_stage_v541(st.model, backbone_trainable=True)
        monitor = (update % monitor_every == 0 or update == 1
                   or update == total)
        positions = _batch_positions_v541(st, update)
        st.schedule.record(update, positions)
        set_diagnostics_enabled_v541(st.model, monitor)
        t_data = time.time()
        batch, records = _batch_from_positions(
            st.dataset, st.lru, positions, st.device,
            prefetcher=st.prefetcher)
        data_s = time.time() - t_data
        t_core = time.time()
        st.optimizer.zero_grad(set_to_none=True)
        loss_out = _loss_on_batch(
            st.model, batch, records, st.labels, st.endpoint, cfg,
            endpoint_pos_weight=st.endpoint_pos_weight,
            pair_rng=_update_rng(int(cfg.seed), int(update), salt=1))
        if bool(loss_out["loss"].requires_grad):
            loss_out["loss"].backward()
            trainable = [p for p in st.model.parameters() if p.requires_grad]
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable,
                                               float(cfg.grad_clip))
            st.optimizer.step()
        core_s = time.time() - t_core
        last_loss = float(loss_out["loss"].detach().cpu())
        st.timers.add(phase, "data_s", data_s)
        st.timers.add(phase, "core_s", core_s)
        st.timers.add(phase, "wall_s", time.time() - t_iter)
        st.timers.count(phase)
        updates_run += 1
        if monitor:
            _log_row(st.log_path, {
                "event": "profile_train",
                "update": int(update),
                "phase": phase,
                "profile_only": True,
                "loss": last_loss,
                "encoder_trainable": any(
                    p.requires_grad for p in st.model.backbone.parameters()),
                "elapsed_s": round(time.time() - st.t_run_start, 3),
            })

    timing = st.timers.as_dict()
    per_update = {}
    for phase in ("profile_w0", "profile_w1"):
        if phase in timing:
            per_update[phase] = round(
                timing[phase]["wall_s"]
                / max(timing[phase]["updates"], 1.0), 6)
    report: Dict[str, Any] = {
        "schema": V541_TRAIN_SCHEMA,
        "profile_only": True,
        "is_formal_run": False,
        "run_outcome": None,
        "run_outcome_reason": ("profile_only_window_is_not_a_formal_run "
                               "(plan §8.3)"),
        "run_outcome_vocabulary": list(RUN_OUTCOMES_V541),
        "objective": cfg.objective,
        "embedding_id": cfg.embedding_id,
        "init_seed": cfg.init_seed,
        "seed": int(cfg.seed),
        "updates_run": int(updates_run),
        "profile_window": {"profile_w0_updates": w0,
                           "profile_w1_updates": w1,
                           "total_updates": total},
        "formal_schedule_untouched": {
            "w0_updates": int(cfg.effective_freeze_updates),
            "w1_updates": int(cfg.joint_updates),
            "total_updates": int(cfg.total_updates),
            "note": ("a profile window NEVER changes the formal "
                     "400 + 37,500 schedule (plan §8.3)"),
        },
        "timing": timing,
        "per_update_mean_s": per_update,
        "resources": {
            "host_peak_rss_bytes": host_peak_rss_bytes(),
            **_gpu_memory_metrics(cfg),
        },
        "cache": {"hits": int(st.stats["cache_hits"]),
                  "misses": int(st.stats["cache_misses"])},
        "fallback_counts": dict(st.stats["fallback"]),
        "batch_schedule_sha256": st.schedule.chain_sha256,
        "evaluation_scope": "none_profile_window",
        "dev_eval_done": False,
        "identity": dict(st.identity),
        "provenance": dict(st.prov),
        "git": dict(st.git),
        "env": dict(st.env),
        "last_loss": last_loss,
        "device": st.device,
        "wall_clock_s": round(time.time() - st.t_run_start, 3),
    }
    (st.out / "profile.json").write_text(
        json.dumps(report, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------


def train_run_v541(dataset: Any, labels_v52: Any, endpoint_labels: Any,
                   config: V541TrainerConfig, out_dir: Any, *,
                   r3_ckpt: Any = None,
                   r3_eval_ckpt: Any = None,
                   dev_positions: Optional[Sequence[int]] = None,
                   train_positions: Optional[Sequence[int]] = None,
                   resume: bool = True,
                   provenance: Optional[Mapping[str, Any]] = None,
                   feature_workers: Optional[int] = None) -> Dict[str, Any]:
    """Train ONE V5.4.1 unified run (or a marked profile window).

    ``dev_positions`` is the frozen dev subset of the split manifest and
    is used ONLY to exclude dev curves from the sampling pool — no dev
    feature, no dev label, no dev evaluation is ever built (WP1 seal).
    The final detailed evaluation runs on the fixed inner-val slice
    (cap 8192) against the matched R3 reference, and the returned metrics
    say so explicitly (``evaluation_scope="inner_val"``,
    ``dev_eval_done=False``).

    ``resume=True`` continues from ``checkpoint_last`` ONLY when its
    identity (all SHAs / embedding / init seed / seed / tier / λ) matches
    this config — otherwise a typed refusal.  The manifest is APPEND-only.
    """
    prefetcher = maybe_create_prefetcher(
        FEATURE_LEVEL_V53MP, False, workers=feature_workers)
    workers = int(prefetcher.workers) if prefetcher is not None else 0
    try:
        return _train_run_v541_impl(
            dataset, labels_v52, endpoint_labels, config, out_dir,
            r3_ckpt=r3_ckpt, r3_eval_ckpt=r3_eval_ckpt,
            dev_positions=dev_positions, train_positions=train_positions,
            resume=resume, provenance=provenance,
            feature_prefetcher=prefetcher, feature_workers=workers)
    finally:
        if prefetcher is not None:
            prefetcher.close()


def _train_run_v541_impl(dataset: Any, labels_v52: Any, endpoint_labels: Any,
                         config: V541TrainerConfig, out_dir: Any, *,
                         r3_ckpt: Any = None,
                         r3_eval_ckpt: Any = None,
                         dev_positions: Optional[Sequence[int]] = None,
                         train_positions: Optional[Sequence[int]] = None,
                         resume: bool = True,
                         provenance: Optional[Mapping[str, Any]] = None,
                         feature_prefetcher: Any = None,
                         feature_workers: int = 0) -> Dict[str, Any]:
    st = _prepare_run(
        dataset, labels_v52, endpoint_labels, config, out_dir,
        r3_ckpt=r3_ckpt, dev_positions=dev_positions,
        train_positions=train_positions, provenance=provenance,
        feature_prefetcher=feature_prefetcher,
        feature_workers=feature_workers,
        resume=bool(resume) and not bool(config.profile_only))

    if config.profile_only:
        report = _run_profile(st)
        (st.out / "metrics.json").write_text(
            json.dumps(report, indent=1, sort_keys=True, default=str) + "\n",
            encoding="utf-8")
        return report

    loop = _run_formal(st)
    return _finalise_run(st, loop, r3_eval_ckpt=r3_eval_ckpt)


def _finalise_run(st: SimpleNamespace, loop: Mapping[str, Any], *,
                  r3_eval_ckpt: Any) -> Dict[str, Any]:
    """Selection → τ → inner-val detailed evaluation → metrics.json."""
    cfg = st.config
    device = st.device
    #: formal-schedule label (§7.4): only the frozen 37,900-update run
    #: with no smoke override — a fixture/200-smoke stays a real run with
    #: an honest ``done`` outcome, but never a formal attempt.
    formal = is_formal_schedule(cfg)

    # --- selection: inner-val ONLY ----------------------------------------
    if st.best_path.is_file():
        state = torch.load(str(st.best_path), map_location="cpu",
                           weights_only=False)
        st.model.load_state_dict(state["model"])
        selected_update = int(state["update"])
    else:                                     # pragma: no cover - defensive
        selected_update = int(cfg.total_updates)
        _save_checkpoint(st.best_path, _new_payload(st, selected_update),
                         st.manifest_path, "checkpoint_best")

    # --- τ_end on inner-val, frozen into BOTH payloads --------------------
    t_tau = time.time()
    tau_block = _calibrate_tau_on(
        st.model, st.dataset, st.val_lru, st.inner_val_pos, st.labels,
        st.endpoint, cfg, endpoint_pos_weight=st.endpoint_pos_weight)
    st.timers.add("eval", "eval_s", time.time() - t_tau)
    for path in (st.best_path, st.last_path):
        if path.is_file():
            payload = torch.load(str(path), map_location="cpu",
                                 weights_only=False)
            payload["tau_end_calibration"] = dict(tau_block)
            payload["content_digest_sha256"] = content_digest_of(payload)
            _save_checkpoint(path, payload, st.manifest_path,
                             "tau_calibration")
    if st.best_path.is_file():
        state = torch.load(str(st.best_path), map_location="cpu",
                           weights_only=False)
        st.model.load_state_dict(state["model"])

    # --- constant-q baseline from the INNER-TRAIN slice only ---------------
    train_q_mean = _train_q_mean(st.samples, st.labels, st.train_pos)

    # --- matched R3 reference (SAME inner-val positions) -------------------
    r3_model = None
    if r3_eval_ckpt is not None:
        payload = torch.load(str(r3_eval_ckpt), map_location="cpu",
                             weights_only=False)
        r3_model = StarEGNNV3()
        r3_model.load_state_dict(payload["model"], strict=True)
        r3_model.to(device)

    set_diagnostics_enabled_v541(st.model, False)  # one-shot eval, no sync
    t_eval = time.time()
    eval_block = evaluate_v53mp(
        st.model, r3_model, st.dataset, st.labels, st.endpoint,
        _CacheView(st.val_lru), list(st.inner_val_pos), cfg,
        train_q_mean=train_q_mean, tau_end=tau_block.get("tau_end"))
    st.timers.add("eval", "eval_s", time.time() - t_eval)
    # the legacy guard text says "dev positions"; under eval_scope=
    # inner_val_only the SAME guard runs on inner-val — state it plainly.
    if isinstance(eval_block.get("move_guard_vs_r3"), dict):
        eval_block["move_guard_vs_r3"]["baseline"] = (
            "r3_eval_ckpt zero-shot on the SAME inner-val positions "
            "(evaluation_scope=inner_val; dev is never evaluated in WP4b)")
        eval_block["move_guard_vs_r3"]["positions_scope"] = "inner_val"

    raw_outcome = eval_block.get("run_outcome", "ok")
    run_outcome = ("done" if raw_outcome == "ok"
                   else (raw_outcome if raw_outcome in RUN_OUTCOMES_V541
                         else "invalid_run"))

    # legacy eval keys are renamed: the positions are the INNER-VAL slice.
    n_eval_samples = eval_block.pop("n_dev_samples", len(st.inner_val_pos))
    n_eval_curves = eval_block.pop("n_dev_curves", len(st.val_curves))
    eval_block.pop("schema", None)

    gpu_mem = _gpu_memory_metrics(cfg)
    timing = st.timers.as_dict()
    metrics: Dict[str, Any] = dict(eval_block)
    metrics.update({
        "schema": V541_TRAIN_SCHEMA,
        "run_outcome": run_outcome,
        "run_outcome_vocabulary": list(RUN_OUTCOMES_V541),
        "objective": cfg.objective,
        "embedding_id": cfg.embedding_id,
        "init_seed": cfg.init_seed,
        "seed": int(cfg.seed),
        "backbone_init": cfg.backbone_init,
        # NOTE: the internal loss arm is NEVER surfaced here — logs,
        # metrics and reports expose ONE objective only (§3.1.4: 新配置/
        # 日志/报表/run_id 不暴露三臂矩阵).
        # --- evaluation scope (§7.2 / WP1 seal) ----------------------------
        "evaluation_scope": "inner_val",
        "eval_scope": cfg.eval_scope,
        "dev_eval_done": False,
        "dev_positions_used_for": "curve_exclusion_only",
        "n_dev_curves_excluded": len(st.dev_curves),
        "dev_curve_ids": sorted(st.dev_curves),
        "confirm_read": False,
        "selection_metric": "inner-val total loss",
        "tau_calibrated_on": "inner_val",
        "n_eval_samples": int(n_eval_samples),
        "n_eval_curves": int(n_eval_curves),
        "n_eval_scope_cap": (None if cfg.inner_val_max_samples is None
                             else int(cfg.inner_val_max_samples)),
        "eval_max_samples_ignored": True,
        # --- identity -------------------------------------------------------
        "identity": dict(st.identity),
        "config_full_sha256": config_full_sha256(cfg),
        # --- schedule / sampler ---------------------------------------------
        "schedule": {
            "objective_mapping": "single unified MOVE+PROCESS loss",
            "w0_updates": int(cfg.effective_freeze_updates),
            "w1_updates": int(cfg.joint_updates),
            "total_updates": int(cfg.total_updates),
            "updates_run": int(loop.get("updates_run", 0)),
            "lr_encoder": float(cfg.lr_encoder),
            "lr_head": float(cfg.lr_head),
            "optimizer": "AdamW",
            "weight_decay": float(cfg.weight_decay),
            "grad_clip": float(cfg.grad_clip),
            "precision": "fp32",
            "lambda_rank": float(cfg.lambda_rank),
            "lambda_q": float(cfg.lambda_q),
            "lambda_endpoint": float(cfg.lambda_endpoint),
            "batch_curves": int(cfg.batch_curves),
            "frames_per_curve": int(cfg.frames_per_curve),
            "val_every": int(cfg.val_every),
            "grad_monitor_every": int(cfg.grad_monitor_every),
            "feature_level": FEATURE_LEVEL_V53MP,
            "profile_only": False,
            "formal_schedule": bool(formal),
        },
        "param_groups": param_group_report_v541(st.model, cfg),
        "sampler": {
            "sampler_id": SAMPLER_ID_V541,
            "spec_sha256": sampler_spec_sha256(),
            "pool_sha256": st.index.pool_sha256,
            "pool_curves": len(st.index.curves),
            "batch_schedule_sha256": st.schedule.chain_sha256,
            "schedule_updates": st.schedule.records,
            "fallback_counts": dict(st.stats["fallback"]),
        },
        # --- selection -------------------------------------------------------
        "selected_update": selected_update,
        "best_inner_val_loss": (None if st.best_val == float("inf")
                                else float(st.best_val)),
        "best_update": int(st.best_update),
        "last_train_loss": loop.get("last_loss"),
        "tau_end_calibration": tau_block,
        # --- position bookkeeping ---------------------------------------------
        "n_train_samples": len(st.train_pos),
        "n_inner_val_samples": len(st.inner_val_pos),
        "n_pool_curves": len(st.pool_curves),
        "train_curves": sorted(st.train_curves),
        "inner_val_curves": sorted(st.val_curves),
        "endpoint_pos_weight": st.ep_weights,
        # --- resource / timing ------------------------------------------------
        "timing": {
            "startup_wall_s": timing.get("startup", {}).get("wall_s", 0.0),
            "phases": timing,
            "eval_wall_s": st.timers.total("eval_s"),
            "checkpoint_wall_s": st.timers.total("checkpoint_s"),
            "wall_clock_s": round(time.time() - st.t_run_start, 3),
            "loop_wall_s": loop.get("loop_wall_s"),
        },
        "resources": {
            "host_peak_rss_bytes": host_peak_rss_bytes(),
            "gpu_max_memory_allocated_bytes":
                gpu_mem["gpu_max_memory_allocated_bytes"],
            "gpu_max_memory_reserved_bytes":
                gpu_mem["gpu_max_memory_reserved_bytes"],
            "gpu_memory_fraction_applied":
                gpu_mem["gpu_memory_fraction_applied"],
            "eval_vram_gate_waits_s": [float(w)
                                       for w in st.eval_gate_waits],
        },
        "cache": {"hits": int(st.stats["cache_hits"]),
                  "misses": int(st.stats["cache_misses"]),
                  "capacity": int(cfg.feature_cache_capacity),
                  "scope": ("this process only — the feature LRU is not "
                            "checkpointed, so a resumed leg recounts its "
                            "own hits/misses"),
                  "hit_rate": round(
                      st.stats["cache_hits"]
                      / max(st.stats["cache_hits"] + st.stats["cache_misses"],
                            1), 6)},
        "device": st.device,
        "feature_workers": int(st.feature_workers),
        "seeds": st.seed_map,
        "git": dict(st.git),
        "env": dict(st.env),
        "provenance": dict(st.prov),
        "profile_only": False,
        "is_formal_run": bool(formal),
        "checkpoint_best_path": str(st.best_path),
        "checkpoint_last_path": str(st.last_path),
        "checkpoint_best_sha256": sha256_file(st.best_path),
        "manifest_path": str(st.manifest_path),
        "head_parameter_counts": st.model.head_parameter_counts(),
        "run_id_namespace": "V541_",
        "ledger_status_written": False,
    })
    (st.out / "metrics.json").write_text(
        json.dumps(metrics, indent=1, sort_keys=True, default=str) + "\n",
        encoding="utf-8")
    append_manifest(st.manifest_path, {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": "run_complete",
        "run_outcome": run_outcome,
        "selected_update": selected_update,
        "best_update": int(st.best_update),
        "batch_schedule_sha256": st.schedule.chain_sha256,
        "config_sha256": st.identity.get("config_sha256"),
        "is_formal_schedule": bool(formal),
        "metrics_path": str(st.out / "metrics.json"),
        "ledger_status_written": False,
    })
    metrics["metrics_path"] = str(st.out / "metrics.json")
    return metrics
