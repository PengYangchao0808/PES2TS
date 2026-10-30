"""V5.4.2 WP3 lane P3-B — the candidate-agnostic trainer/runner core
(``v542_train.py``; ADR-0033 §2.6, config ``training_contract``, plan §6
/ §8.2).

Discipline: this module REUSES the frozen V5.3-MP/V5.4.1 machinery by
importing it — the loss, the sampler binding, the optimizer conventions,
the RNG/digest/manifest helpers and the checkpoint save path are CALLED,
never re-implemented and never forked:

* **loss** — ``_loss_on_batch`` (the frozen ``key_atom_loss_v53mp`` with
  the internal arm-C λ set ``λr=0.5, λq=0.5, λe=0.012385049119773344``)
  for training and ``_inner_val_loss`` for selection; pair draws stay
  the pure ``_update_rng(seed, update, salt)`` protocol (salt 1 = train,
  salt 2 = inner-val);
* **sampler** — the frozen V541 common sampler binding: ``StrataIndex``
  built ONCE, per-update draws through ``curve_batch_positions_v541`` /
  ``batch_rng_v541``, the ``BatchScheduleHasher`` chain, the fallback
  counters, and a construction-time guard that the LIVE
  ``sampler_spec_sha256()`` still equals the config-pinned
  ``training_contract.sampler_binding.spec_sha256`` (typed
  ``V542_SAMPLER_DRIFT`` refusal otherwise);
* **optimizer** — AdamW (``weight_decay`` from config) over the ASSEMBLY's
  by-``id()`` param groups: ``core`` at ``lr 1e-4`` (backbone ∪
  move_decoder) and ``head`` at ``lr 3e-4``; the lrs are frozen (typed
  ``V542_LR_FROZEN`` refusal on drift);
* **checkpoint/resume** — V541's atomic ``_save_checkpoint`` + the
  ``v541_identity`` payload machinery (python/numpy/torch RNG states,
  content digest, APPEND-only manifest), re-expressed for V542 payloads;
* **identity** — the caller's identity SHA block plus the canonical
  run-identity fields (seed, candidate, λ, config SHA, sampler
  id/spec/pool) form ONE identity block whose SHA-256 digest must match
  on resume.

Two phases, frozen (config ``param_groups`` / ``schedule``):

======  ==============================  ==========================
phase   updates                         trainable groups
======  ==============================  ==========================
``w0``  ``1 … freeze_encoder_updates``  head live, core FROZEN
``w1``  ``… + joint_updates``           ALL joint (core + head)
======  ==============================  ==========================

"Core frozen" means ``requires_grad=False`` ONLY — the forward is NEVER
wrapped in ``no_grad()`` (frozen ≠ no_grad, ADR-0033 §2.1 / plan §6).

Guarantees of this core:

* **append-only run records** — ``run_records.jsonl`` gets EXACTLY one
  row per checkpoint / inner-val event, each carrying ``step``,
  ``inner_val_total_loss``, ``wall_s`` / ``core_s``, ``seed`` and the
  identity SHA fields (``identity`` block + ``identity_sha256``);
  ``train_log.jsonl`` and ``manifest.jsonl`` are appended, never
  rewritten;
* **selection is inner-val ONLY** — ``checkpoint_best.pt`` is chosen by
  the inner-val total loss at ``val_every`` cadence; H2/H3 metrics are
  never computed, never stored and can never select a checkpoint;
* **crash-safe resume** — only from ``checkpoint_last.pt``, only when
  the identity DIGEST matches (a mismatch — including a seed switch — is
  a typed ``invalid_run`` refusal raised BEFORE anything is written, so
  the checkpoint is left byte-for-byte untouched); the restore brings
  back model / optimizer / phase / step / all RNG states / sampler
  counter / schedule chain;
* **typed outcomes** — ``gate_skipped`` / ``invalid_run`` /
  ``resource_failure`` (plus ``budget_censored``) are operational
  outcomes, structurally distinct from scientific results (``done`` /
  ``scientific_failure``).

Scope: trainer core only — no H0/H1 gate execution, no ledger writes, no
GPU formal run, no QC, no dev/confirm access, no campaign runner.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

from pes2ts_core.pesn.models.v542.assembly import (
    LR_CORE_V542,
    LR_HEAD_V542,
    AssembledV542Model,
    assemble_v542_model,
)
from pes2ts_core.pesn.models.v542.identity import canonical_sha256
from pes2ts_core.pesn.models.v542.interfaces import N_PROCESS_PARAMS_V542
from pes2ts_core.pesn.offline.training.key_atom_bc_v3 import (
    set_seed,
    sha256_file,
)
from pes2ts_core.pesn.offline.training.key_atom_v53mp_train_v3 import (
    FEATURE_LEVEL_V53MP,
    PROVENANCE_PLACEHOLDER_KEYS,
    V53MPTrainerConfig,
    _FeatureLRU,
    _assert_label_coverage,
    _assert_no_confirm,
    _batch_from_positions,
    _curve_id_set_of,
    _inner_val_loss,
    _loss_on_batch,
    _reset_peak_memory_stats,
    _startup_of,
    _update_rng,
    _wait_for_eval_vram_headroom,
    apply_cuda_memory_fraction,
    estimate_endpoint_pos_weight,
    v53mp_pool_split,
)
from pes2ts_core.pesn.offline.training.v541_identity import (
    RUN_OUTCOMES_V541,
    append_manifest,
    content_digest_of,
    env_state,
    git_state,
    rng_state_restore,
    rng_state_snapshot,
    run_outcome_for_exception,
)
from pes2ts_core.pesn.offline.training.v541_monitor import PhaseTimer
from pes2ts_core.pesn.offline.training.v541_strata import (
    SAMPLER_ID_V541,
    BatchScheduleHasher,
    StrataIndex,
    batch_rng_v541,
    curve_batch_positions_v541,
    sampler_spec_sha256,
)
from pes2ts_core.pesn.offline.training.v541_train import (
    V541TrainerError,     # the typed V541 refusal this trainer extends
    _log_rows,            # generic append-JSONL-lines helper (route E)
    _positions_block,     # generic data/shard position block (route E)
    _save_checkpoint,     # atomic save + APPEND-only manifest line
)

__all__ = [
    "OPERATIONAL_OUTCOMES_V542",
    "RUN_OUTCOMES_V542",
    "SCIENTIFIC_OUTCOMES_V542",
    "V542_FROZEN_SEEDS",
    "V542_FORMAL_TOTAL_UPDATES",
    "V542_LAMBDA_ENDPOINT",
    "V542_LAMBDA_Q",
    "V542_LAMBDA_RANK",
    "V542_SAMPLER_ID",
    "V542_SAMPLER_SPEC_SHA256_PINNED",
    "V542_TRAIN_SCHEMA",
    "V542_W0_UPDATES",
    "V542_W1_UPDATES",
    "V542Outcome",
    "V542Trainer",
    "V542TrainerConfig",
    "V542TrainerError",
    "assert_resume_identity",
    "build_checkpoint_v542",
    "config_full_sha256",
    "config_identity_dict",
    "config_identity_sha256",
    "gate_skipped",
    "grad_finite_state",
    "identity_block_v542",
    "invalid_run",
    "is_formal_schedule_v542",
    "make_optimizer_v542",
    "outcome_for_exception",
    "param_group_report_v542",
    "param_groups_v542",
    "resource_failure",
    "set_training_stage_v542",
    "train_run_v542",
]

#: Checkpoint / identity schema id of this trainer.
V542_TRAIN_SCHEMA: str = "pesn_v542_train_v1"

#: Frozen λ of the unified objective (config ``training_contract.lambdas``).
V542_LAMBDA_RANK: float = 0.5
V542_LAMBDA_Q: float = 0.5
V542_LAMBDA_ENDPOINT: float = 0.012385049119773344

#: Frozen schedule (config ``training_contract.schedule``).
V542_W0_UPDATES: int = 400
V542_W1_UPDATES: int = 37500
V542_FORMAL_TOTAL_UPDATES: int = V542_W0_UPDATES + V542_W1_UPDATES   # 37,900

#: The frozen run seeds — also the process-head anchor seeds (config
#: ``init_policy.process_head_anchor.seeds``).
V542_FROZEN_SEEDS: tuple = (17, 29, 43)

#: Frozen sampler binding of V5.4.2 = the V541 common sampler (config
#: ``training_contract.sampler_binding``: the live spec SHA must equal
#: this pinned value, else typed ``V542_SAMPLER_DRIFT``).
V542_SAMPLER_ID: str = SAMPLER_ID_V541
V542_SAMPLER_SPEC_SHA256_PINNED: str = (
    "0769817ef01dca3abbb1d67f5405e98138b4707e74b690240d3f9b6821b65d9e")

#: Six-word outcome vocabulary — inherited verbatim from V541 §7.4 and
#: kept separate from the legacy run-ledger status machine.
RUN_OUTCOMES_V542: tuple = RUN_OUTCOMES_V541
#: Outcomes a COMPLETED scientific run can carry.
SCIENTIFIC_OUTCOMES_V542: tuple = ("done", "scientific_failure")
#: Operational outcomes — never scientific results, always reason-coded.
OPERATIONAL_OUTCOMES_V542: tuple = (
    "invalid_run", "resource_failure", "budget_censored", "gate_skipped")

#: Config fields that do NOT change what one training update computes
#: (run-length / wall-clock knobs) and therefore stay OUT of the resume
#: identity — a 3-update leg must be able to resume into a 6-update run,
#: while seed / λ / lr / batch / val cadence / identity SHAs all remain
#: inside the digest.
_V542_IDENTITY_EXCLUDED_FIELDS = frozenset({
    "smoke_updates",
    "cuda_memory_fraction",
    "eval_vram_gate_mib",
    "eval_vram_gate_timeout_s",
})


class V542TrainerError(V541TrainerError):
    """V5.4.2 trainer input/state violation (typed, fail-closed).

    ``fields`` names the offending identity keys on a resume refusal;
    ``outcome`` is the six-word outcome this refusal maps to (default
    ``invalid_run``), so operational failures stay structurally distinct
    from scientific results.
    """

    def __init__(self, code: str, detail: str,
                 fields: Optional[Sequence[str]] = None, *,
                 outcome: str = "invalid_run") -> None:
        super().__init__(code, detail, fields=fields)
        if outcome not in RUN_OUTCOMES_V542:
            raise ValueError(  # pragma: no cover - programmer error
                f"outcome {outcome!r} not in {RUN_OUTCOMES_V542}")
        self.outcome = str(outcome)


def _require(cond: bool, code: str, detail: str,
             fields: Optional[Sequence[str]] = None, *,
             outcome: str = "invalid_run") -> None:
    if not cond:
        raise V542TrainerError(code, detail, fields=fields, outcome=outcome)


# ---------------------------------------------------------------------------
# typed outcomes (six-word vocabulary; operational ≠ scientific)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class V542Outcome:
    """A six-word run outcome plus its reason code.

    ``gate_skipped`` / ``invalid_run`` / ``resource_failure`` (and
    ``budget_censored``) are OPERATIONAL: they carry a mandatory reason
    code and are NEVER scientific results (``done`` /
    ``scientific_failure``).  The distinction is structural:
    :attr:`is_scientific_result` is computed from the frozen vocabulary,
    so an operational outcome can never be reported as a result.
    """

    outcome: str
    reason: str = ""

    def __post_init__(self) -> None:
        if self.outcome not in RUN_OUTCOMES_V542:
            raise V542TrainerError(
                "OUTCOME_VOCABULARY",
                f"outcome {self.outcome!r} not in the frozen six-word "
                f"vocabulary {RUN_OUTCOMES_V542}")
        if self.outcome in OPERATIONAL_OUTCOMES_V542 and \
                not str(self.reason).strip():
            raise V542TrainerError(
                "OUTCOME_REASON",
                f"operational outcome {self.outcome!r} must carry a "
                "non-empty reason code (plan §12)")

    @property
    def is_scientific_result(self) -> bool:
        """True ONLY for ``done`` / ``scientific_failure``."""
        return self.outcome in SCIENTIFIC_OUTCOMES_V542


def gate_skipped(reason: str) -> V542Outcome:
    """``gate_skipped(reason_code)`` — scheduling decision, 0 new training."""
    return V542Outcome("gate_skipped", str(reason))


def invalid_run(reason: str) -> V542Outcome:
    """``invalid_run(reason)`` — protocol/identity/implementation failure."""
    return V542Outcome("invalid_run", str(reason))


def resource_failure(reason: str) -> V542Outcome:
    """``resource_failure(reason)`` — OOM/host-limit class failure."""
    return V542Outcome("resource_failure", str(reason))


def outcome_for_exception(exc: BaseException) -> V542Outcome:
    """Map any exception to a typed :class:`V542Outcome`.

    Reuses the frozen ``v541_identity.run_outcome_for_exception``
    mapping (explicit ``exc.outcome`` wins; OOM → ``resource_failure``;
    everything else → ``invalid_run``) and attaches the exception text
    as the mandatory operational reason.
    """
    name = run_outcome_for_exception(exc)
    reason = "{}: {}".format(type(exc).__name__, exc)
    return V542Outcome(name, reason)


# ---------------------------------------------------------------------------
# config (frozen V5.3-MP/V541 knobs + the V5.4.2 identity surface)
# ---------------------------------------------------------------------------
@dataclass
class V542TrainerConfig(V53MPTrainerConfig):
    """One planned V5.4.2 run (config ``training_contract``).

    Subclasses the frozen V5.3-MP config so every loss / sampler /
    optimizer knob keeps ONE definition, then pins the V5.4.2 surface:
    external ``objective="unified_move_process"`` (the internal arm is
    always ``C`` — all three λ > 0 — and is never surfaced in logs or
    metrics), the frozen λ set, the frozen 400 + 37,500 schedule, the
    frozen run seeds ``{17, 29, 43}``, R3-only warm start, the frozen
    by-id lrs and the candidate id that :func:`assemble_v542_model`
    builds.
    """

    # -- internal loss arm (C only) + external objective -------------------
    arm: str = "C"
    seed: int = 17
    objective: str = "unified_move_process"
    # -- frozen λ of the unified objective ---------------------------------
    lambda_rank: float = V542_LAMBDA_RANK
    lambda_q: float = V542_LAMBDA_Q
    lambda_endpoint: float = V542_LAMBDA_ENDPOINT
    # -- frozen schedule (config training_contract.schedule) ---------------
    freeze_encoder_updates: int = V542_W0_UPDATES
    joint_updates: int = V542_W1_UPDATES
    # -- frozen inner-val cap (config split_contract.inner_val_max_samples) --
    inner_val_max_samples: Optional[int] = 8192
    # -- V5.4.2 identity ---------------------------------------------------
    candidate_id: str = "A0"
    #: Selection/evaluation scope: this core only ever touches inner-val
    #: (dev is curve-exclusion only; confirm is never read).
    eval_scope: str = "inner_val_only"

    def __post_init__(self) -> None:
        super().__post_init__()               # arm/λ/schedule rules (frozen)
        _require(self.objective == "unified_move_process", "OBJECTIVE",
                 "objective must be 'unified_move_process', got "
                 "{!r}".format(self.objective))
        _require(self.eval_scope == "inner_val_only", "EVAL_SCOPE",
                 "eval_scope must be 'inner_val_only', got "
                 "{!r}".format(self.eval_scope))
        _require(isinstance(self.candidate_id, str)
                 and bool(self.candidate_id), "CANDIDATE_ID",
                 "candidate_id must be a non-empty string, got "
                 "{!r}".format(self.candidate_id))
        _require(int(self.seed) in V542_FROZEN_SEEDS, "V542_SEED_FROZEN",
                 "run seed must be one of the frozen seeds {} (config "
                 "process_head_anchor.seeds — the process-head anchor seed "
                 "IS the run seed), got {!r}".format(
                     list(V542_FROZEN_SEEDS), self.seed))
        _require(self.backbone_init == "r3", "V542_BACKBONE_INIT",
                 "V5.4.2 warm-start is R3-only (config backbone_init), got "
                 "{!r}".format(self.backbone_init))
        _require(float(self.lr_encoder) == LR_CORE_V542
                 and float(self.lr_head) == LR_HEAD_V542,
                 "V542_LR_FROZEN",
                 "frozen by-id lrs are core={} / head={}; got lr_encoder="
                 "{} lr_head={} (config init_policy.param_groups — no "
                 "schedule changes beyond the frozen ones)".format(
                     LR_CORE_V542, LR_HEAD_V542,
                     self.lr_encoder, self.lr_head))

    # -- schedule aliases ---------------------------------------------------
    @property
    def w0_updates(self) -> int:
        """Frozen-core window length (updates 1…400 on a formal run)."""
        return int(self.effective_freeze_updates)

    @property
    def w1_updates(self) -> int:
        """Frozen joint window length (updates 401…37,900 on a formal run)."""
        return int(self.joint_updates)


def is_formal_schedule_v542(config: V542TrainerConfig) -> bool:
    """Preregistered-schedule label: ONLY the frozen 37,900-update run
    with no ``smoke_updates`` override is a formal attempt (a smoke /
    fixture run stays a real run with an honest ``done`` outcome)."""
    return (config.smoke_updates is None
            and int(config.total_updates) == V542_FORMAL_TOTAL_UPDATES)


# ---------------------------------------------------------------------------
# identity block + resume refusal
# ---------------------------------------------------------------------------
def config_identity_dict(config: V542TrainerConfig) -> Dict[str, Any]:
    """Resume-identity projection of the config: ``asdict`` minus the
    run-length / wall-clock knobs of :data:`_V542_IDENTITY_EXCLUDED_FIELDS`
    (they change WHEN a run stops, never WHAT one update computes)."""
    blob = asdict(config)
    for key in _V542_IDENTITY_EXCLUDED_FIELDS:
        blob.pop(key, None)
    return blob


def config_identity_sha256(config: V542TrainerConfig) -> str:
    """Canonical SHA of :func:`config_identity_dict` (the ONE strict
    recipe — ``identity.canonical_sha256``, never ``default=str``)."""
    return canonical_sha256(config_identity_dict(config))


def config_full_sha256(config: V542TrainerConfig) -> str:
    """Canonical SHA of the WHOLE config (provenance record)."""
    return canonical_sha256(asdict(config))


def identity_block_v542(config: V542TrainerConfig,
                        identity: Optional[Mapping[str, Any]], *,
                        pool_sha256: Optional[str]) -> Dict[str, Any]:
    """The ONE identity block stored in every checkpoint and compared on
    resume.

    The caller's identity SHA block (``source_manifest`` / config /
    protocol / environment / init-mapping / anchor / statistics / plan
    SHAs … — the runner lane's product) is carried VERBATIM, then the
    canonical run-identity fields are written on top (canonical keys win,
    so a caller cannot smuggle a different seed past this seam): schema,
    objective, candidate id, run seed, the anchor seed (IS the run
    seed), the three frozen λ, the R3-only backbone init, the config
    identity SHA and the frozen sampler id / spec / pool SHAs.
    """
    block: Dict[str, Any] = dict(identity or {})
    block.update({
        "schema": V542_TRAIN_SCHEMA,
        "objective": str(config.objective),
        "candidate_id": str(config.candidate_id),
        "seed": int(config.seed),
        "anchor_seed": int(config.seed),
        "backbone_init": str(config.backbone_init),
        "lambda_rank": float(config.lambda_rank),
        "lambda_q": float(config.lambda_q),
        "lambda_endpoint": float(config.lambda_endpoint),
        "eval_scope": str(config.eval_scope),
        "config_sha256": config_identity_sha256(config),
        "sampler_id": V542_SAMPLER_ID,
        "sampler_spec_sha256": sampler_spec_sha256(),
        "sampler_pool_sha256": pool_sha256,
    })
    return block


def assert_resume_identity(
        state: Mapping[str, Any],
        config: V542TrainerConfig,
        identity: Optional[Mapping[str, Any]], *,
        pool_sha256: Optional[str]) -> Dict[str, Any]:
    """Typed refusal unless ``checkpoint_last`` matches THIS identity.

    Compares the full identity block key-by-key (offending keys land on
    ``exc.fields``) AND its SHA-256 digest.  Every mismatch — a switched
    seed, a changed λ, a drifted source-manifest SHA, a foreign schema —
    raises ``RESUME_IDENTITY_MISMATCH``, which maps to the operational
    outcome ``invalid_run``: resume NEVER silently starts a fresh run and
    a crashed run can never switch seeds.
    """
    _require(isinstance(state, Mapping), "RESUME_INVALID",
             "checkpoint_last payload is not a mapping")
    _require(state.get("schema") == V542_TRAIN_SCHEMA,
             "RESUME_IDENTITY_MISMATCH",
             "checkpoint schema {!r} != {!r}".format(
                 state.get("schema"), V542_TRAIN_SCHEMA),
             fields=["schema"])
    expected = identity_block_v542(config, identity, pool_sha256=pool_sha256)
    stored = state.get("identity")
    _require(isinstance(stored, Mapping), "RESUME_IDENTITY_MISMATCH",
             "checkpoint carries no identity block", fields=["identity"])
    stored = dict(stored)
    sentinel = object()
    mismatches = sorted(
        key for key in set(expected) | set(stored)
        if expected.get(key, sentinel) != stored.get(key, sentinel))
    if mismatches:
        detail = "; ".join(
            "{}: checkpoint={!r} run={!r}".format(
                k, stored.get(k, "<absent>"), expected.get(k, "<absent>"))
            for k in mismatches)
        raise V542TrainerError(
            "RESUME_IDENTITY_MISMATCH",
            "checkpoint_last belongs to a different run — " + detail,
            fields=mismatches)
    expected_digest = canonical_sha256(expected)
    stored_digest = state.get("identity_sha256")
    _require(stored_digest == expected_digest
             and canonical_sha256(stored) == stored_digest,
             "RESUME_IDENTITY_MISMATCH",
             "identity digest mismatch (stored={!r}, expected={!r})".format(
                 stored_digest, expected_digest),
             fields=["identity_sha256"])
    return stored


# ---------------------------------------------------------------------------
# param groups / optimizer / phase switching (by id — never by name)
# ---------------------------------------------------------------------------
def param_groups_v542(assembled: AssembledV542Model,
                      config: V542TrainerConfig) -> List[Dict[str, Any]]:
    """The two FROZEN optimizer groups straight from the assembly record.

    ``core`` = backbone ∪ move_decoder at ``lr_encoder = 1e-4`` and
    ``head`` = process_head at ``lr_head = 3e-4`` — grouped BY ``id()``
    inside :func:`assemble_v542_model`, so the shared ``w_v`` sits in
    exactly ONE group and can never be double-updated.  Fresh dicts are
    returned every call (``torch.optim`` mutates the dicts it is given).
    """
    _require(float(config.lr_encoder) == LR_CORE_V542
             and float(config.lr_head) == LR_HEAD_V542,
             "V542_LR_FROZEN",
             "config lrs drifted from the frozen core/head lrs")
    return assembled.optimizer_param_groups()


def make_optimizer_v542(
        assembled: AssembledV542Model,
        config: V542TrainerConfig) -> torch.optim.Optimizer:
    """AdamW (``weight_decay`` from the frozen config) over the two by-id
    groups — the same optimizer convention as the V5.3-MP/V541 trainers."""
    return torch.optim.AdamW(
        param_groups_v542(assembled, config),
        weight_decay=float(config.weight_decay))


def set_training_stage_v542(assembled: AssembledV542Model, *,
                            core_trainable: bool) -> None:
    """Switch phases by ``requires_grad`` ONLY (W0: core frozen, head
    live; W1: all joint).  The forward is NEVER put under ``no_grad()`` —
    that would sever the process head's gradient path (frozen ≠ no_grad).
    """
    for param in assembled.core_params:
        param.requires_grad_(bool(core_trainable))
    for param in assembled.head_params:
        param.requires_grad_(True)


def param_group_report_v542(
        assembled: AssembledV542Model) -> List[Dict[str, Any]]:
    """JSON-safe by-id group summary for ``metrics.json``."""
    return [{"name": entry["name"], "lr": float(entry["lr"]),
             "n_params": int(sum(int(p.numel()) for p in entry["params"])),
             "n_tensors": len(entry["params"])}
            for entry in assembled.param_groups]


def grad_finite_state(model: torch.nn.Module) -> Dict[str, Any]:
    """Trainer-side gradient-finiteness screen (oracle gate-3 C3).

    Returns the FROZEN record fields the campaign harness consumes (name
    frozen by ``v542_campaign.TRAINER_GRAD_FINITE_FIELD`` /
    ``TRAINER_GRAD_NONFINITE_PARAMS_FIELD``):

    * ``grad_finite_all_finite`` — ``True`` iff every parameter that has
      a gradient has an all-finite one (a ``None`` grad — a frozen or
      never-touched parameter — is vacuously finite);
    * ``grad_nonfinite_params`` — the SORTED names of the offending
      parameters, present ONLY when the flag is ``False``.

    The check runs over live tensors (``torch.isfinite``), never over a
    serialized snapshot, so a NaN/Inf injected into a gradient by a bad
      batch flips the flag on the very next record row.
    """
    nonfinite: List[str] = []
    for name, param in model.named_parameters():
        grad = param.grad
        if grad is None:
            continue
        if not bool(torch.isfinite(grad).all()):
            nonfinite.append(str(name))
    if nonfinite:
        return {"grad_finite_all_finite": False,
                "grad_nonfinite_params": sorted(nonfinite)}
    return {"grad_finite_all_finite": True}


def _assert_sampler_binding() -> None:
    """The live ``v541_strata`` spec MUST still be the config-pinned one.

    Config ``training_contract.sampler_binding`` freezes V5.4.2 to the
    V541 common sampler (``sampler_id`` + ``spec_sha256``); a drift
    between the pinned values and the live module is a typed
    ``V542_SAMPLER_DRIFT`` refusal — the run must not start on a
    different sampler than it was preregistered with.
    """
    _require(V542_SAMPLER_ID == SAMPLER_ID_V541, "V542_SAMPLER_DRIFT",
             "sampler id {!r} != {!r}".format(
                 V542_SAMPLER_ID, SAMPLER_ID_V541))
    live = sampler_spec_sha256()
    _require(live == V542_SAMPLER_SPEC_SHA256_PINNED,
             "V542_SAMPLER_DRIFT",
             "live sampler_spec_sha256 {} != config-pinned {}".format(
                 live, V542_SAMPLER_SPEC_SHA256_PINNED))


# ---------------------------------------------------------------------------
# checkpoint payload (model / optimizer / phase / step / RNG / sampler /
# identity digest) — V541 §7.3 shape, V542 schema
# ---------------------------------------------------------------------------
def build_checkpoint_v542(
        *, assembled: AssembledV542Model, optimizer: torch.optim.Optimizer,
        update: int, phase: str, best_val: float, best_update: int,
        config: V542TrainerConfig, seeds: Mapping[str, Any],
        identity: Mapping[str, Any], provenance: Mapping[str, Any],
        schedule: BatchScheduleHasher,
        sampler_counters: Mapping[str, int],
        data_position: Mapping[str, Any],
        git: Mapping[str, Any], env: Mapping[str, Any],
) -> Dict[str, Any]:
    """One atomic checkpoint payload — every required field, digest last."""
    payload: Dict[str, Any] = {
        "schema": V542_TRAIN_SCHEMA,
        "model": assembled.model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "update": int(update),
        "phase": str(phase),
        "best_val": float(best_val),
        "best_update": int(best_update),
        "config": asdict(config),
        "config_full_sha256": config_full_sha256(config),
        "identity": dict(identity),
        "identity_sha256": canonical_sha256(dict(identity)),
        "provenance": dict(provenance),
        "seeds": dict(seeds),
        "candidate_id": str(config.candidate_id),
        "param_counts": {"backbone_active": int(assembled.active_params),
                         "joint": int(assembled.joint_params)},
        # --- RNG (python / numpy / torch / torch_cuda) --------------------
        "rng": rng_state_snapshot(config.device),
        # --- sampler / schedule -------------------------------------------
        "sampler": {
            "sampler_id": V542_SAMPLER_ID,
            "spec_sha256": sampler_spec_sha256(),
            "pool_sha256": identity.get("sampler_pool_sha256"),
            "batch_curves": int(config.batch_curves),
            "frames_per_curve": int(config.frames_per_curve),
            "counters": dict(sampler_counters),
        },
        "schedule": schedule.state_dict(),
        "sampler_counters": dict(sampler_counters),
        # --- data / shard position + provenance ---------------------------
        "data_position": dict(data_position),
        "git": dict(git),
        "env": dict(env),
    }
    payload["content_digest_sha256"] = content_digest_of(payload)
    return payload


# ---------------------------------------------------------------------------
# append-only records (run_records.jsonl / train_log.jsonl / manifest.jsonl)
# ---------------------------------------------------------------------------
def _run_record(st: SimpleNamespace, event: str, step: int, phase: str,
                **extra: Any) -> Dict[str, Any]:
    """Append ONE run-record row (the required per checkpoint/eval audit
    row: step, inner-val total loss, wall/core seconds, seed, the
    identity SHA fields AND the trainer-side gradient-finiteness screen
    ``grad_finite_all_finite`` — oracle gate-3 C3, the field the campaign
    diagnostic harness consumes; ``grad_nonfinite_params`` appears only
    when the flag is False).  APPEND-only — the file is never rewritten."""
    row: Dict[str, Any] = {
        "event": str(event),
        "step": int(step),
        "phase": str(phase),
        "objective": st.config.objective,
        "candidate_id": st.config.candidate_id,
        "seed": int(st.config.seed),
        "wall_s": round(time.time() - st.t_run_start, 3),
        "core_s": round(st.timers.total("core_s"), 3),
        "identity": dict(st.identity),
        "identity_sha256": st.identity_digest,
        "grad_finite_all_finite": bool(
            st.grad_finite.get("grad_finite_all_finite", True)),
    }
    if not row["grad_finite_all_finite"]:
        row["grad_nonfinite_params"] = list(
            st.grad_finite.get("grad_nonfinite_params", []))
    row.update(extra)
    return append_manifest(st.records_path, row)


def _stage_switch_row(st: SimpleNamespace, update: int,
                      phase: str) -> Dict[str, Any]:
    """W0 → W1 evidence: the core is IN the optimizer and gradients reach
    it after the switch (and NO core grad exists while frozen)."""
    core = list(st.assembled.core_params)
    grads = [p.grad for p in core if p.grad is not None]
    max_abs = (max(float(g.abs().max()) for g in grads)
               if grads else None)
    return {
        "event": "stage_switch",
        "update": int(update),
        "phase": phase,
        "core_requires_grad": all(p.requires_grad for p in core),
        "n_core_params": len(core),
        "n_core_params_with_grad": len(grads),
        "core_grad_max_abs": max_abs,
        "core_reachable": bool(grads) and bool(
            max_abs is not None and max_abs > 0.0),
        "note": ("frozen != no_grad: the forward ran WITH graph, so the "
                 "process head stays differentiable in W0 and the core "
                 "starts receiving gradients in W1 (plan §6)"),
    }


def _train_row(st: SimpleNamespace, update: int, phase: str, *,
               loss_out: Mapping[str, Any], positions: Sequence[int],
               stage_switch: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """One per-update ``train_log.jsonl`` row (append-only)."""
    cfg = st.config
    row: Dict[str, Any] = {
        "event": "train",
        "update": int(update),
        "phase": phase,
        "objective": cfg.objective,
        "candidate_id": cfg.candidate_id,
        "seed": int(cfg.seed),
        "loss": float(loss_out["loss"].detach().cpu()),
        "l_move": float(loss_out["l_move"].detach().cpu()),
        "l_rank": float(loss_out["l_rank"].detach().cpu()),
        "l_q": float(loss_out["l_q"].detach().cpu()),
        "l_endpoint": float(loss_out["l_endpoint"].detach().cpu()),
        "n_rank_pairs": int(loss_out["n_rank_pairs"]),
        "n_q_rows": int(loss_out["n_q_rows"]),
        "n_endpoint_rows": int(loss_out["n_endpoint_rows"]),
        "batch_positions": [int(p) for p in positions],
        "sampler_counters": dict(st.counters),
        "core_trainable": all(p.requires_grad
                              for p in st.assembled.core_params),
        "head_trainable": all(p.requires_grad
                              for p in st.assembled.head_params),
        "wall_s": round(time.time() - st.t_run_start, 3),
        "core_s": round(st.timers.total("core_s"), 3),
        "elapsed_s": round(time.time() - st.t_run_start, 3),
    }
    if stage_switch is not None:
        row["stage_switch"] = dict(stage_switch)
    return row


# ---------------------------------------------------------------------------
# the trainer
# ---------------------------------------------------------------------------
class V542Trainer:
    """Two-phase (W0/W1) V5.4.2 training core over one assembled model.

    One instance == one planned run: ``run()`` prepares the deterministic
    position/sampler/model/identity state (assembling the model through
    :func:`~pes2ts_core.pesn.models.v542.assembly.assemble_v542_model`),
    executes updates ``start_update … total_updates`` with the frozen
    loss/sampler/optimizer, selects ``checkpoint_best.pt`` on the
    inner-val total loss ONLY, appends its records and returns a
    ``metrics.json``-shaped dict with ``run_outcome="done"``.

    Args:
        config: the frozen-surface :class:`V542TrainerConfig`.
        out_dir: run directory (``checkpoint_{last,best}.pt``,
            ``run_records.jsonl``, ``train_log.jsonl``,
            ``manifest.jsonl``, ``metrics.json``).
        dataset / labels_v52 / endpoint_labels: the SAME dataset surface
            the frozen V5.3-MP/V541 trainers consume (records → features
            → real collate/forward-view batches).
        dev_positions: frozen dev subset — used ONLY to EXCLUDE dev
            curves from the sampling pool (no dev feature/label/eval).
        train_positions: optional explicit FULL pool (train + inner-val);
            ``None`` derives it by dev-curve exclusion.
        identity: run-identity SHA block recorded verbatim and digested
            into the resume identity.
        require_identity: opt into the strict assembly identity check
            (absent/dirty identity becomes a typed refusal).
        resume: continue from ``checkpoint_last.pt`` when present — only
            when its identity block AND digest match; a mismatch raises
            before anything is written.
    """

    def __init__(self, *, config: V542TrainerConfig, out_dir: Any,
                 dataset: Any, labels_v52: Any, endpoint_labels: Any,
                 dev_positions: Sequence[int],
                 train_positions: Optional[Sequence[int]] = None,
                 provenance: Optional[Mapping[str, Any]] = None,
                 identity: Optional[Mapping[str, Any]] = None,
                 require_identity: bool = False,
                 resume: bool = True) -> None:
        self.config = config
        self.out_dir = Path(out_dir)
        self.dataset = dataset
        self.labels_v52 = labels_v52
        self.endpoint_labels = endpoint_labels
        self.dev_positions = [int(p) for p in (dev_positions or ())]
        self.train_positions = (None if train_positions is None
                                else [int(p) for p in train_positions])
        self.provenance = dict(provenance or {})
        self.identity_in = (None if identity is None else dict(identity))
        self.require_identity = bool(require_identity)
        self.resume = bool(resume)
        self.assembled: Optional[AssembledV542Model] = None
        self.st: Optional[SimpleNamespace] = None

    # -- public entry -------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Prepare → train → finalise; returns the ``metrics.json`` dict."""
        st = self._prepare()
        self.st = st                      # inspectable even if the loop fails
        loop = self._train_loop(st)
        return self._finalise(st, loop)

    # -- deterministic setup (mirrors the frozen V541 ``_prepare_run``) -----
    def _prepare(self) -> SimpleNamespace:
        cfg = self.config
        out = Path(self.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        _assert_sampler_binding()

        prov: Dict[str, Any] = {k: None for k in PROVENANCE_PLACEHOLDER_KEYS}
        prov.update(dict(self.provenance))
        seed_map = set_seed(int(cfg.seed))
        device = cfg.device
        if str(device).startswith("cuda") and not torch.cuda.is_available():
            raise V542TrainerError("CUDA_UNAVAILABLE",
                                   "device=cuda but CUDA is not available")
        apply_cuda_memory_fraction(cfg)
        _reset_peak_memory_stats(device)

        t_startup = time.time()
        timers = PhaseTimer()

        # --- position sets (fail-closed BEFORE any model/loss work) --------
        samples = self.dataset.samples
        n_samples = len(samples)
        _require(bool(self.dev_positions), "DEV_POSITIONS",
                 "dev_positions must be non-empty — they exist to EXCLUDE "
                 "the dev curves from the pool (never to be evaluated here)")
        dev_pos = list(self.dev_positions)
        for p in dev_pos:
            _require(0 <= p < n_samples, "POSITION_RANGE",
                     "position {} outside 0..{}".format(p, n_samples - 1))
        dev_curve_set = _curve_id_set_of(samples, dev_pos)

        if self.train_positions is None:
            startup = _startup_of(samples)
            if startup is not None:
                dev_rows = np.asarray(
                    sorted(startup.curve_row[c] for c in dev_curve_set
                           if c in startup.curve_row), dtype=np.int64)
                pool = np.flatnonzero(
                    ~np.isin(np.asarray(startup.curve_idx), dev_rows)
                ).astype(np.int64).tolist()
            else:
                pool = [i for i in range(n_samples)
                        if str(samples[i].curve_id) not in dev_curve_set]
        else:
            pool = list(self.train_positions)
        for p in pool:
            _require(0 <= p < n_samples, "POSITION_RANGE",
                     "position {} outside 0..{}".format(p, n_samples - 1))
        pool_curves = _curve_id_set_of(samples, pool)
        train_pos, inner_val_pos = v53mp_pool_split(
            samples, pool_curve_ids=sorted(pool_curves), seed=int(cfg.seed),
            inner_val_fraction=float(cfg.inner_val_fraction))
        if self.train_positions is not None and \
                set(pool) != set(train_pos) | set(inner_val_pos):
            raise V542TrainerError(
                "TRAIN_POOL_NOT_CURVE_COMPLETE",
                "train_positions must be the FULL pool (train + inner-val "
                "curves) returned by v53mp_pool_split; got {} positions, the "
                "curve-level split covers {}".format(
                    len(pool), len(set(train_pos) | set(inner_val_pos))))
        _assert_no_confirm(samples, train_pos, "train")
        _assert_no_confirm(samples, inner_val_pos, "inner-val")
        _assert_no_confirm(samples, dev_pos, "dev")

        # --- fixed inner-val cap (selection only; frozen 8192 default) -----
        if cfg.inner_val_max_samples is not None and \
                len(inner_val_pos) > int(cfg.inner_val_max_samples):
            rng_iv = np.random.RandomState(int(cfg.seed) ^ 0x1E7A1)
            inner_val_pos = sorted(rng_iv.choice(
                list(inner_val_pos), size=int(cfg.inner_val_max_samples),
                replace=False).tolist())

        # --- the frozen stratified sampler index (built ONCE) --------------
        index = StrataIndex.build(samples, train_pos, dataset=self.dataset)
        train_curves = _curve_id_set_of(samples, train_pos)
        val_curves = _curve_id_set_of(samples, inner_val_pos)
        dev_curves = _curve_id_set_of(samples, dev_pos)
        _require(not (dev_curves & (train_curves | val_curves)),
                 "DEV_TRAIN_OVERLAP",
                 "dev curves overlap the train pool: {}".format(
                     sorted(dev_curves & (train_curves | val_curves))))
        _require(bool(index.curves), "SAMPLER_POOL_EMPTY",
                 "StrataIndex covers no inner-train curve")

        # --- label coverage over train + inner-val ONLY --------------------
        _assert_label_coverage(samples, self.labels_v52, self.endpoint_labels,
                               [train_pos, inner_val_pos], n_samples)
        ep_weights = estimate_endpoint_pos_weight(
            samples, self.endpoint_labels, train_pos, seed=int(cfg.seed))
        endpoint_pos_weight = float(ep_weights["pos_weight"])

        # --- model: assembled AFTER the deterministic setup (V5.2/V541
        # discipline); the assembly runs steps 1→5 under fork_rng and step 6
        # (finish_training_seed) resets the global RNG to the run seed ------
        set_seed(int(cfg.seed))
        assembled = assemble_v542_model(
            cfg.candidate_id, int(cfg.seed), device=device,
            identity=self.identity_in,
            require_identity=self.require_identity)
        self.assembled = assembled
        set_training_stage_v542(
            assembled,
            core_trainable=(1 > cfg.effective_freeze_updates))
        optimizer = make_optimizer_v542(assembled, cfg)

        identity = identity_block_v542(cfg, self.identity_in,
                                       pool_sha256=index.pool_sha256)
        identity_digest = canonical_sha256(identity)
        git = git_state()
        env = env_state()

        # --- resume (identity-checked BEFORE anything is written) ----------
        last_path = out / "checkpoint_last.pt"
        best_path = out / "checkpoint_best.pt"
        manifest_path = out / "manifest.jsonl"
        records_path = out / "run_records.jsonl"
        log_path = out / "train_log.jsonl"
        start_update = 1
        best_val = float("inf")
        best_update = 0
        schedule = BatchScheduleHasher()
        counters: Dict[str, int] = {}
        if self.resume and last_path.is_file():
            state = torch.load(str(last_path), map_location="cpu",
                               weights_only=False)
            assert_resume_identity(state, cfg, self.identity_in,
                                   pool_sha256=index.pool_sha256)
            stored_digest = state.get("content_digest_sha256")
            actual_digest = content_digest_of(state)
            if not isinstance(stored_digest, str) or \
                    actual_digest != stored_digest:
                raise V542TrainerError(
                    "RESUME_CONTENT_DIGEST",
                    "checkpoint_last content digest mismatch — payload was "
                    "modified after it was written (stored={!r}, "
                    "recomputed={!r}); refusing to load".format(
                        stored_digest, actual_digest),
                    fields=["content_digest_sha256"])
            assembled.model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            start_update = int(state["update"]) + 1
            best_val = float(state["best_val"])
            best_update = int(state["best_update"])
            resumed_phase = str(state.get("phase", ""))
            expected_phase = ("w0" if int(state["update"])
                              <= cfg.effective_freeze_updates else "w1")
            _require(resumed_phase == expected_phase, "RESUME_PHASE_MISMATCH",
                     "checkpoint phase {!r} != {!r} for update {}".format(
                         resumed_phase, expected_phase,
                         int(state["update"])),
                     fields=["phase"])
            schedule = BatchScheduleHasher.from_state(state["schedule"])
            counters = dict(state.get("sampler_counters") or {})
            rng_state_restore(state["rng"])     # python / numpy / torch (+cuda)
            append_manifest(manifest_path, {
                "ts": datetime.now(timezone.utc).isoformat(),
                "event": "resume",
                "from_update": int(state["update"]),
                "next_update": int(start_update),
                "phase": resumed_phase,
                "identity_sha256": identity_digest,
                "content_digest_sha256": stored_digest,
            })
        set_training_stage_v542(
            assembled,
            core_trainable=(start_update > cfg.effective_freeze_updates))

        lru = _FeatureLRU(self.dataset,
                          capacity=int(cfg.feature_cache_capacity),
                          level=FEATURE_LEVEL_V53MP, intent_off=False)
        val_lru = _FeatureLRU(self.dataset,
                              capacity=max(len(inner_val_pos), 1) + 64,
                              level=FEATURE_LEVEL_V53MP, intent_off=False)

        seed_record: Dict[str, Any] = dict(seed_map)
        seed_record.update({
            "run_seed": int(cfg.seed),
            "anchor_seed": int(assembled.anchor_seed),
            "seed_init": int(assembled.seed_init),
            "init_source_id": str(assembled.init_source_id),
        })
        timers.add("startup", "wall_s", time.time() - t_startup)
        return SimpleNamespace(
            out=out, config=cfg, prov=prov, seed_record=seed_record,
            device=device, dataset=self.dataset, labels=self.labels_v52,
            endpoint=self.endpoint_labels, samples=samples,
            n_samples=n_samples, train_pos=train_pos,
            inner_val_pos=inner_val_pos, dev_pos=dev_pos,
            dev_curves=dev_curves, train_curves=train_curves,
            val_curves=val_curves, pool_curves=pool_curves, index=index,
            ep_weights=ep_weights, endpoint_pos_weight=endpoint_pos_weight,
            lru=lru, val_lru=val_lru, assembled=assembled,
            model=assembled.model, optimizer=optimizer, schedule=schedule,
            counters=counters, identity=identity,
            identity_digest=identity_digest, git=git, env=env,
            timers=timers, eval_gate_waits=[],
            grad_finite={"grad_finite_all_finite": True},
            log_path=log_path, records_path=records_path,
            last_path=last_path, best_path=best_path,
            manifest_path=manifest_path, start_update=start_update,
            best_val=best_val, best_update=best_update,
            t_run_start=time.time())

    # -- the frozen two-phase loop -----------------------------------------
    def _train_loop(self, st: SimpleNamespace) -> Dict[str, Any]:
        cfg = st.config
        total = int(cfg.total_updates)
        t_loop = time.time()
        last_loss: Optional[float] = None
        updates_run = 0

        for update in range(st.start_update, total + 1):
            t_iter = time.time()
            phase = "w0" if update <= cfg.effective_freeze_updates else "w1"
            stage_switch: Optional[Dict[str, Any]] = None
            stage_switch_pending = (
                update == cfg.effective_freeze_updates + 1)
            if stage_switch_pending:
                # W0 → W1: the core joins the optimizer from this update on.
                set_training_stage_v542(st.assembled, core_trainable=True)

            # --- sampler draw (pure (seed, update, 0)) + schedule chain ----
            positions = curve_batch_positions_v541(
                st.index, rng=batch_rng_v541(int(cfg.seed), int(update)),
                batch_curves=int(cfg.batch_curves), counters=st.counters)
            st.schedule.record(int(update), positions)

            # --- data lane (real records → features → real collate) --------
            t_data = time.time()
            batch, records = _batch_from_positions(
                st.dataset, st.lru, positions, st.device, prefetcher=None)
            data_s = time.time() - t_data

            # --- core lane: frozen loss, backward, clip, step --------------
            t_core = time.time()
            st.optimizer.zero_grad(set_to_none=True)
            loss_out = _loss_on_batch(
                st.model, batch, records, st.labels, st.endpoint, cfg,
                endpoint_pos_weight=st.endpoint_pos_weight,
                pair_rng=_update_rng(int(cfg.seed), int(update), salt=1))
            if bool(loss_out["loss"].requires_grad):
                loss_out["loss"].backward()
                # trainer-side gradient finiteness screen (gate-3 C3):
                # measured on the RAW backward output, every update, so
                # every record row carries a fresh grad_finite_all_finite
                st.grad_finite = grad_finite_state(st.model)
                trainable = [p for p in st.model.parameters()
                             if p.requires_grad]
                if trainable:
                    torch.nn.utils.clip_grad_norm_(
                        trainable, float(cfg.grad_clip))
                st.optimizer.step()
            else:
                _log_rows(st.log_path, [{
                    "event": "skipped_no_grad",
                    "update": int(update),
                    "reason": ("loss has no grad path (no active supervised "
                               "rows in this batch)")}])
            core_s = time.time() - t_core

            last_loss = float(loss_out["loss"].detach().cpu())
            st.timers.add(phase, "data_s", data_s)
            st.timers.add(phase, "core_s", core_s)
            st.timers.add(phase, "wall_s", time.time() - t_iter)
            st.timers.count(phase)
            updates_run += 1

            if stage_switch_pending:
                stage_switch = _stage_switch_row(st, update, phase)
                _log_rows(st.log_path, [stage_switch])
            _log_rows(st.log_path, [
                _train_row(st, update, phase, loss_out=loss_out,
                           positions=positions, stage_switch=stage_switch)])

            # --- inner-val sweep: selection on inner-val total loss ONLY ----
            if update % max(int(cfg.val_every), 1) == 0 or update == total:
                _wait_for_eval_vram_headroom(cfg, st.eval_gate_waits)
                t_eval = time.time()
                val = _inner_val_loss(
                    st.model, st.dataset, st.val_lru, st.inner_val_pos,
                    st.labels, st.endpoint, cfg,
                    endpoint_pos_weight=st.endpoint_pos_weight,
                    update=update, prefetcher=None)
                st.timers.add(phase, "eval_s", time.time() - t_eval)
                improved = val < st.best_val
                if improved:
                    st.best_val = float(val)
                    st.best_update = int(update)
                _run_record(st, "inner_val", update, phase,
                            inner_val_total_loss=float(val),
                            best_inner_val_total_loss=float(st.best_val),
                            best_update=int(st.best_update),
                            selection_metric="inner-val total loss")
                if improved:
                    t_ck = time.time()
                    _save_checkpoint(
                        st.best_path,
                        self._payload(st, update, phase),
                        st.manifest_path, "checkpoint_best")
                    st.timers.add(phase, "checkpoint_s",
                                  time.time() - t_ck)
                    _run_record(st, "checkpoint_best", update, phase,
                                inner_val_total_loss=float(val),
                                checkpoint="checkpoint_best.pt",
                                best_update=int(st.best_update),
                                best_inner_val_total_loss=float(st.best_val))
                t_ck = time.time()
                _save_checkpoint(
                    st.last_path,
                    self._payload(st, update, phase),
                    st.manifest_path, "checkpoint_last")
                st.timers.add(phase, "checkpoint_s", time.time() - t_ck)
                _run_record(st, "checkpoint_last", update, phase,
                            inner_val_total_loss=float(val),
                            checkpoint="checkpoint_last.pt",
                            best_update=int(st.best_update),
                            best_inner_val_total_loss=float(st.best_val))

        return {"last_loss": last_loss, "total_updates": total,
                "updates_run": int(updates_run),
                "loop_wall_s": round(time.time() - t_loop, 3)}

    def _payload(self, st: SimpleNamespace, update: int,
                 phase: str) -> Dict[str, Any]:
        return build_checkpoint_v542(
            assembled=st.assembled, optimizer=st.optimizer,
            update=update, phase=phase, best_val=st.best_val,
            best_update=st.best_update, config=st.config,
            seeds=st.seed_record, identity=st.identity,
            provenance=st.prov, schedule=st.schedule,
            sampler_counters=st.counters,
            data_position=_positions_block(
                st.train_pos, st.inner_val_pos, st.dev_curves, st.prov,
                st.n_samples),
            git=st.git, env=st.env)

    # -- selection → metrics (inner-val ONLY; H2/H3 never computed) ---------
    def _finalise(self, st: SimpleNamespace,
                  loop: Mapping[str, Any]) -> Dict[str, Any]:
        cfg = st.config
        if st.best_path.is_file():
            state = torch.load(str(st.best_path), map_location="cpu",
                               weights_only=False)
            st.model.load_state_dict(state["model"])
            selected_update = int(state["update"])
        else:                                     # pragma: no cover - defensive
            selected_update = int(cfg.total_updates)
            _save_checkpoint(st.best_path,
                             self._payload(st, selected_update,
                                           "w1" if selected_update >
                                           cfg.effective_freeze_updates
                                           else "w0"),
                             st.manifest_path, "checkpoint_best")

        metrics: Dict[str, Any] = {
            "schema": V542_TRAIN_SCHEMA,
            "run_outcome": "done",
            "run_outcome_vocabulary": list(RUN_OUTCOMES_V542),
            "objective": cfg.objective,
            "candidate_id": cfg.candidate_id,
            "seed": int(cfg.seed),
            "anchor_seed": int(cfg.seed),
            "is_formal_schedule": is_formal_schedule_v542(cfg),
            "updates_run": int(loop.get("updates_run", 0)),
            "last_train_loss": loop.get("last_loss"),
            # --- selection: inner-val total loss ONLY (plan §8.2) ----------
            "selection_metric": "inner-val total loss",
            "h3_metric_selection_forbidden": True,
            "selected_update": selected_update,
            "best_inner_val_loss": (None if st.best_val == float("inf")
                                    else float(st.best_val)),
            "best_update": int(st.best_update),
            "evaluation_scope": "inner_val",
            "eval_scope": cfg.eval_scope,
            "dev_eval_done": False,
            "dev_positions_used_for": "curve_exclusion_only",
            "confirm_read": False,
            # --- identity ---------------------------------------------------
            "identity": dict(st.identity),
            "identity_sha256": st.identity_digest,
            "config_sha256": config_identity_sha256(cfg),
            "config_full_sha256": config_full_sha256(cfg),
            # --- schedule / optimizer (frozen) ------------------------------
            "schedule": {
                "w0_updates": int(cfg.effective_freeze_updates),
                "w1_updates": int(cfg.joint_updates),
                "total_updates": int(cfg.total_updates),
                "updates_run": int(loop.get("updates_run", 0)),
                "lr_core": float(cfg.lr_encoder),
                "lr_head": float(cfg.lr_head),
                "optimizer": "AdamW",
                "weight_decay": float(cfg.weight_decay),
                "grad_clip": float(cfg.grad_clip),
                "precision": "fp32",
                "torch_compile": False,
                "lambda_rank": float(cfg.lambda_rank),
                "lambda_q": float(cfg.lambda_q),
                "lambda_endpoint": float(cfg.lambda_endpoint),
                "batch_curves": int(cfg.batch_curves),
                "frames_per_curve": int(cfg.frames_per_curve),
                "val_every": int(cfg.val_every),
                "loop_wall_s": loop.get("loop_wall_s"),
            },
            "param_groups": param_group_report_v542(st.assembled),
            "param_counts": {
                "backbone_active": int(st.assembled.active_params),
                "decoder": (int(st.assembled.joint_params)
                            - int(st.assembled.active_params)
                            - int(N_PROCESS_PARAMS_V542)),
                "process_head": int(N_PROCESS_PARAMS_V542),
                "joint": int(st.assembled.joint_params),
            },
            "param_group_by_id": True,
            "shared_w_v_single_update": True,
            # --- sampler (frozen V541 binding) -----------------------------
            "sampler": {
                "sampler_id": V542_SAMPLER_ID,
                "spec_sha256": sampler_spec_sha256(),
                "spec_sha256_pinned": V542_SAMPLER_SPEC_SHA256_PINNED,
                "pool_sha256": st.index.pool_sha256,
                "pool_curves": len(st.index.curves),
                "batch_schedule_sha256": st.schedule.chain_sha256,
                "schedule_updates": st.schedule.records,
                "fallback_counters": dict(st.counters),
            },
            "seeds": st.seed_record,
            "param_counts_formula_note": (
                "decoder = joint - active - process_head"),
            "timing": {
                "startup_wall_s": st.timers.as_dict()
                .get("startup", {}).get("wall_s", 0.0),
                "phases": st.timers.as_dict(),
                "eval_wall_s": st.timers.total("eval_s"),
                "checkpoint_wall_s": st.timers.total("checkpoint_s"),
                "wall_clock_s": round(time.time() - st.t_run_start, 3),
            },
            "checkpoint_best_path": str(st.best_path),
            "checkpoint_last_path": str(st.last_path),
            "checkpoint_best_sha256": sha256_file(st.best_path),
            "run_records_path": str(st.records_path),
            "train_log_path": str(st.log_path),
            "manifest_path": str(st.manifest_path),
            "git": dict(st.git),
            "env": dict(st.env),
            "provenance": dict(st.prov),
            "run_id_namespace": "V542_",
            "ledger_status_written": False,
            "qc_calls": 0,
        }
        (st.out / "metrics.json").write_text(
            json.dumps(metrics, indent=1, sort_keys=True, default=str) + "\n",
            encoding="utf-8")
        append_manifest(st.manifest_path, {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "run_complete",
            "run_outcome": "done",
            "selected_update": selected_update,
            "best_update": int(st.best_update),
            "batch_schedule_sha256": st.schedule.chain_sha256,
            "identity_sha256": st.identity_digest,
            "metrics_path": str(st.out / "metrics.json"),
            "ledger_status_written": False,
        })
        metrics["metrics_path"] = str(st.out / "metrics.json")
        return metrics


# ---------------------------------------------------------------------------
# public entry point (mirrors ``train_run_v541``)
# ---------------------------------------------------------------------------
def train_run_v542(dataset: Any, labels_v52: Any, endpoint_labels: Any,
                   config: V542TrainerConfig, out_dir: Any, *,
                   dev_positions: Sequence[int],
                   train_positions: Optional[Sequence[int]] = None,
                   provenance: Optional[Mapping[str, Any]] = None,
                   identity: Optional[Mapping[str, Any]] = None,
                   require_identity: bool = False,
                   resume: bool = True) -> Dict[str, Any]:
    """Train ONE V5.4.2 run (or continue it) and write the run artifacts.

    ``dev_positions`` is the frozen dev subset — used ONLY to exclude dev
    curves from the sampling pool (no dev feature/label/eval is ever
    built).  ``resume=True`` continues from ``checkpoint_last.pt`` only
    when its identity block AND digest match this run — otherwise a
    typed ``invalid_run`` refusal with the offending keys on
    ``exc.fields`` (the checkpoint is left untouched).
    """
    trainer = V542Trainer(
        config=config, out_dir=out_dir, dataset=dataset,
        labels_v52=labels_v52, endpoint_labels=endpoint_labels,
        dev_positions=dev_positions, train_positions=train_positions,
        provenance=provenance, identity=identity,
        require_identity=require_identity, resume=resume)
    return trainer.run()

