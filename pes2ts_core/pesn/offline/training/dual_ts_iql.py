"""Dual-TS IQL trainer semantics (WP5-prep, plan v4 §I.11/§I.12; contract v2 §3–§4).

RESEARCH-ONLY PREPARATION LANE: this module implements the semantics and
their fixture tests.  It must NEVER be pointed at real data before the WP6
gate (BC frozen + RL eligibility).  No monitor/test consumption, no ACP.

v1 defects fixed here (plan §I.11; contract v2 §4):

1. Q receives the PHYSICAL action (Å) through ``dual_ts_critics`` with
   version-flagged units — never the dimensionless latent.
2. Independent Q1/Q2/V networks (each owns its encoder; init digests
   recorded); no weight sharing with the BC actor.
3. BC actor is loaded by BUNDLE IDENTITY before anything trains: the
   checkpoint's real content digest must equal
   ``bundle.bc_parent_checkpoint_digest``, and the features-spec digest,
   action version, and seed must match this run — any mismatch is a typed
   refusal.  The actor stays FROZEN through warmup.
4. Phase machine: ``warmup`` (frozen default 2,000 Q/V updates, zero
   policy updates — the actor is not even forwarded) → ``joint`` (at most
   20,000 updates); transitions and counters are checkpointed.
5. Frozen IQL targets (contract §4): ``q̄ = min(Q̄1, Q̄2)`` (target nets,
   stop-grad); expectile τ = 0.7 V regression; Q target
   ``r + γ · bootstrap_mask · V(s_next)`` (stop-grad); advantage
   ``A = stop_grad(q̄ − V)``; AWR weights ``w = min(exp(A / 3), 100)``
   (3 is the temperature denominator, not a multiplier); target Q nets
   advance by a REGISTERED polyak.
6. SAME §I.10 sampling + log-prob reduction as BC — imported from
   ``dual_ts_sampling`` (single shared surface, no divergence).
7. Multi-episode indexing: episode-keyed buffers; a reaction may carry
   many episodes; successor resolution is restricted to the SAME episode
   (cross-episode chaining is a typed construction refusal); per-head
   (Q/V/actor) eligibility masks come from the DualTSTransition records,
   ineligible rows are skipped and counted.
8. Checkpoint/resume: actor/Q1/Q2/V + target nets, optimizers, phase and
   update counters, sampler state, CPU + CUDA RNG, frozen norm/scale
   params, and all content digests (bundle: ``pesn_dual_ts_bundle_v2``);
   the checkpoint carries its own real content digest and refuses
   tampered loads.  Resuming continues the uninterrupted trajectory
   bit-for-bit on the same seed (fixture test).
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.contracts.dual_ts import (
    ACTION_SCHEMA_VERSION,
    BUNDLE_SCHEMA_VERSION,
    DualTSBundle,
    DualTSTransition,
)
from pes2ts_core.pesn.models.dual_ts_critics import (
    Q_ACTION_INPUT_VERSION,
    S_A_RULE,
    DualTSQCritic,
    DualTSVCritic,
    model_state_digest,
)
from pes2ts_core.pesn.models.dual_egnn import (
    DualTSModelConfig,
    collate_dual_states,
)
from pes2ts_core.pesn.models.dual_ts_policy import (
    DEFAULT_A_MAX_ANGSTROM,
    DualTSPolicy,
)
from pes2ts_core.pesn.models import require_torch
from pes2ts_core.pesn.offline.training.device import (
    move_batch,
    resolve_device,
    seed_everything,
    set_cpu_threads,
)
from pes2ts_core.pesn.offline.training.checkpoints import (
    BC_CHECKPOINT_SCHEMA_V2,
    checkpoint_content_digest,
)
from pes2ts_core.pesn.offline.training.dual_ts_sampling import (
    GROUP_BRANCH_SAMPLING_RULE,
    LOGPROB_REDUCTION_RULE,
    WITHIN_BRANCH_RULE,
    DualTSGroupBranchSampler,
    ExposureTracker,
    build_sampler_groups,
    hierarchical_logprob_reduction,
)

DUAL_TS_IQL_VERSION: Final[str] = "pesn_dual_ts_iql_v2.1"
CHECKPOINT_SCHEMA: Final[str] = "pesn_dual_ts_iql_checkpoint_v2"
BC_CHECKPOINT_SCHEMA: Final[str] = BC_CHECKPOINT_SCHEMA_V2
PHASE_WARMUP: Final[str] = "warmup"
PHASE_JOINT: Final[str] = "joint"
HEADS: Final[tuple[str, ...]] = ("Q", "V", "actor")
#: Registered polyak target pairs (contract §4: 目标网络更新 registered).
POLYAK_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("q1", "q1_target"), ("q2", "q2_target"))


class DualTSIQLTrainerError(RuntimeError):
    """Typed trainer / buffer refusal."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code


class BCActorIdentityError(RuntimeError):
    """Typed BC-actor identity gate refusal (contract v2 §4)."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code


# ---------------------------------------------------------------------------
# Pure frozen formulas (unit-tested directly)
# ---------------------------------------------------------------------------


def expectile_loss(torch, diff, tau: float = 0.7):
    """Expectile-τ regression on ``diff`` (mean reduction, unmasked).

    weight = τ for diff ≥ 0 (over-prediction side), (1 − τ) otherwise.
    """
    w = torch.where(diff >= 0,
                    torch.full_like(diff, float(tau)),
                    torch.full_like(diff, float(1.0 - tau)))
    return (w * diff.pow(2)).mean()


def awr_weights(torch, advantage, temperature: float = 3.0,
                clip: float = 100.0):
    """``w = min(exp(A / T_A), 100)`` with A = stop_grad(q̄ − V).

    The advantage is DETACHED here (contract: stop-gradient); the returned
    weights never carry gradient.  ``temperature`` is the denominator T_A.
    """
    a = advantage.detach()
    w = torch.exp(a / float(temperature))
    return w.clamp(max=float(clip))


# ---------------------------------------------------------------------------
# Canonical content digests over checkpoint payloads
#
# Shared with the BC v2 checkpoint writer via
# :mod:`pes2ts_core.pesn.offline.training.checkpoints` (single source of
# truth); re-exported here for the WP5-prep import surface.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Episode-keyed buffer (multi-episode per reaction, no cross-episode chains)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DualTSTransitionRow:
    """One in-memory transition with its bound state inputs + known reward.

    ``state_inputs`` uses the frozen shard layout consumed by
    ``collate_dual_states`` (REQUIRED_STATE_KEYS).  ``reward`` is the known
    per-transition reward (None + ``reward_missing_reason`` when unknown —
    never a fabricated 0).  ``next_state_inputs`` is optional and only
    cross-checked; the bootstrap state always comes from the buffer's own
    SAME-EPISODE successor row.
    """

    reaction_group_id: str
    transition: DualTSTransition
    state_inputs: Mapping[str, np.ndarray]
    reward: float | None
    reward_missing_reason: str | None = None
    next_state_inputs: Mapping[str, np.ndarray] | None = None
    #: remaining mass-weighted arc length (teacher label, NOT a state input);
    #: optional — used only for the v4.1 exposure accounting near-TS stratum
    #: (``ral <= 0.2 * branch_max``).  None → near-TS unknown (counted as not
    #: near-TS).  Populated by WP6 from the bound teacher table.
    remaining_arc_length: float | None = None


class EpisodeKeyedBuffer:
    """Episode-keyed store of transition rows with per-head eligibility.

    Chaining rule ``successor_resolution_restricted_to_same_episode_v0``:
    a row's ``next_state_id`` must resolve to a state of the SAME episode
    inside this buffer; a reference into another episode (or nowhere) is a
    typed construction refusal.  Terminal rows carry no successor.  This
    is what makes per-reaction many-episode indexing safe: TD chains never
    cross an episode boundary by construction.
    """

    EPISODE_CHAINING_RULE: Final[str] = (
        "successor_resolution_restricted_to_same_episode_v0")

    def __init__(self, rows: Sequence[DualTSTransitionRow]) -> None:
        if not rows:
            raise DualTSIQLTrainerError("EMPTY_BUFFER",
                                        "episode-keyed buffer needs rows")
        self.rows: list[DualTSTransitionRow] = list(rows)
        self.episodes: dict[str, list[int]] = {}
        state_owner: dict[str, str] = {}
        for idx, row in enumerate(self.rows):
            t = row.transition
            t.validate()
            n_atoms = int(np.asarray(row.state_inputs["atomic_numbers"]).size)
            carrier = t.proposed_action.action_angstrom
            if carrier is None:
                raise DualTSIQLTrainerError(
                    "ACTION_CARRIER_MISSING",
                    f"row {t.transition_id} has no physical action carrier")
            if np.asarray(carrier).shape != (n_atoms, 3):
                raise DualTSIQLTrainerError(
                    "ACTION_SHAPE_MISMATCH",
                    f"row {t.transition_id} action "
                    f"{np.asarray(carrier).shape} != atoms {(n_atoms, 3)}")
            if row.reward is None and not row.reward_missing_reason:
                raise DualTSIQLTrainerError(
                    "REWARD_SEMANTICS_INVALID",
                    f"row {t.transition_id}: unknown reward needs a reason")
            if t.state_id in state_owner and \
                    state_owner[t.state_id] != t.episode_id:
                raise DualTSIQLTrainerError(
                    "STATE_ID_REUSED_ACROSS_EPISODES",
                    f"state {t.state_id!r} claimed by episodes "
                    f"{state_owner[t.state_id]!r} and {t.episode_id!r}")
            state_owner.setdefault(t.state_id, t.episode_id)
            self.episodes.setdefault(t.episode_id, []).append(idx)
        # successor resolution STRICTLY within the same episode
        self._next_of: dict[int, int | None] = {}
        for ep, idxs in self.episodes.items():
            local_states = {self.rows[i].transition.state_id: i
                            for i in idxs}
            for i in idxs:
                t = self.rows[i].transition
                if t.next_state_id is None:
                    if not t.terminal:
                        raise DualTSIQLTrainerError(
                            "DANGLING_NON_TERMINAL",
                            f"row {t.transition_id} has no successor but is "
                            "not terminal")
                    self._next_of[i] = None
                    continue
                nxt = local_states.get(t.next_state_id)
                if nxt is None:
                    in_other = t.next_state_id in state_owner
                    raise DualTSIQLTrainerError(
                        "CROSS_EPISODE_SUCCESSOR" if in_other
                        else "SUCCESSOR_UNRESOLVED",
                        f"row {t.transition_id}: successor "
                        f"{t.next_state_id!r} "
                        + ("belongs to a DIFFERENT episode — cross-episode "
                           "chaining is refused"
                           if in_other else
                           "is not present in its episode chain"))
                if self.rows[nxt].transition.episode_id != ep:
                    raise DualTSIQLTrainerError(
                        "CROSS_EPISODE_SUCCESSOR",
                        f"row {t.transition_id} chains across episodes")
                if self.rows[i].next_state_inputs is not None:
                    nxt_atoms = int(np.asarray(
                        self.rows[nxt].state_inputs["atomic_numbers"]).size)
                    given = int(np.asarray(
                        self.rows[i].next_state_inputs[
                            "atomic_numbers"]).size)
                    if given != nxt_atoms:
                        raise DualTSIQLTrainerError(
                            "NEXT_STATE_INPUTS_MISMATCH",
                            f"row {t.transition_id}: next_state_inputs atom "
                            f"count {given} != successor {nxt_atoms}")
                self._next_of[i] = nxt
        # reaction-group membership must be constant per episode
        for ep, idxs in self.episodes.items():
            groups = {self.rows[i].reaction_group_id for i in idxs}
            if len(groups) != 1:
                raise DualTSIQLTrainerError(
                    "EPISODE_SPANS_REACTION_GROUPS",
                    f"episode {ep!r} spans reaction groups {sorted(groups)}")

    def next_row_of(self, row_index: int) -> int | None:
        return self._next_of[int(row_index)]

    def head_groups(self) -> dict[str, dict[str, list[int]]]:
        """§I.10 sampler view: group → branch → row indices (arc order).

        Rows are ordered by ``remaining_arc_length`` ASCENDING (near-TS
        first) — the SAME canonical within-branch order BC uses — so BC and
        IQL produce the IDENTICAL same-seed draw sequence on real data
        (finding 3).  When ``remaining_arc_length`` is missing (None) the
        arc key falls back to (episode_id, chain position) — this is a
        fixture-only path; formal runs refuse missing ral (finding 5).  Built
        via the SHARED :func:`build_sampler_groups` (P06).

        Tie-break edge case (registered): equal ``remaining_arc_length``
        within a branch falls back to ``(episode_id, chain_position)``, which
        may diverge from BC's original record-index tie-break; a multi-episode
        branch with EXACT ral ties could draw in a different order.  The
        current fit1 shard has no ral ties (60/60 holds).
        """
        pos: dict[int, tuple] = {}
        for ep in sorted(self.episodes):
            for k, idx in enumerate(self.episodes[ep]):
                pos[int(idx)] = (ep, k)

        def arc_key(r, i):
            ral = r.remaining_arc_length
            ep, k = pos[int(i)]
            if ral is None:
                # missing ral: fall back to chain position (a distinct sort
                # class so None rows never interleave with ral rows).
                return (1, 0.0, ep, k)
            return (0, float(ral), ep, k)

        return build_sampler_groups(
            self.rows,
            group_key=lambda r: r.reaction_group_id,
            branch_key=lambda r: r.transition.branch_id,
            arc_key=arc_key)

    def _chain_position(self, ep: str, idx: int) -> int:
        return self.episodes[ep].index(idx)


# ---------------------------------------------------------------------------
# Frozen settings
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DualTSIQLSettings:
    """Frozen IQL hyperparameters (plan §I.11 training order; contract §4).

    ``warmup_updates`` / ``joint_max_updates`` carry the plan's frozen
    2,000 / 20,000 defaults; smoke runs MAY shorten them but must record
    the actual phase lengths (they travel inside every checkpoint).
    """

    warmup_updates: int = 2_000
    joint_max_updates: int = 20_000
    batch_reactions: int = 8
    rows_per_group_per_draw: int = 1
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    seed: int = 42
    expectile: float = 0.7
    advantage_temperature: float = 3.0
    weight_clip: float = 100.0
    target_polyak: float = 0.995
    gamma: float = 1.0
    a_max_angstrom: float = DEFAULT_A_MAX_ANGSTROM
    s_a_angstrom: float | None = None     # None → a_max (S_A_RULE)
    sigma_min: float = 0.02
    hidden: int = 128
    n_layers: int = 4
    d_model: int = 32
    d_element: int = 16
    light: bool = False
    signed_mu_gate: bool = False
    device: str = "cuda:0"
    require_cuda: bool = False
    cpu_intra_threads: int = 1
    cpu_inter_threads: int = 1
    log_every: int = 200
    #: v4.1 finding-5 flag: a NON-diagnostic (formal) IQL run refuses if any
    #: transition row carries ``remaining_arc_length=None`` (near-TS integrity);
    #: a diagnostic run instead warns and reports ``near_ts_fraction`` as None.
    diagnostic_run: bool = False

    def to_payload(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}

    @classmethod
    def from_payload(cls, payload: Mapping) -> "DualTSIQLSettings":
        fields = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in dict(payload).items() if k in fields})


# ---------------------------------------------------------------------------
# BC actor loading by bundle identity (contract v2 §4)
# ---------------------------------------------------------------------------


def load_bc_actor(torch, *, policy: DualTSPolicy, payload: Mapping,
                  expected_features_spec_digest: str,
                  expected_action_version: str, expected_seed: int,
                  expected_checkpoint_digest: str) -> dict:
    """Load the matching BC actor or refuse with a typed identity error.

    Gates, in order: registered schema → self content digest consistency →
    bundle binding (``bc_parent_checkpoint_digest``) → features spec digest
    → action version → seed → strict state load.  The recomputed digest is
    a REAL content digest over the serialized payload (tensors included).
    """
    if payload.get("schema_version") != BC_CHECKPOINT_SCHEMA:
        raise BCActorIdentityError(
            "BC_SCHEMA_UNKNOWN",
            f"expected {BC_CHECKPOINT_SCHEMA!r}, got "
            f"{payload.get('schema_version')!r}")
    recorded = payload.get("content_digest")
    if not isinstance(recorded, str) or not recorded:
        raise BCActorIdentityError(
            "BC_DIGEST_MISSING",
            "BC checkpoint carries no real content digest")
    recomputed = checkpoint_content_digest(torch, payload)
    if recomputed != recorded:
        raise BCActorIdentityError(
            "BC_SELF_DIGEST_MISMATCH",
            f"recorded {recorded[:12]}… != recomputed {recomputed[:12]}… "
            "(checkpoint content was modified after saving)")
    if recorded != expected_checkpoint_digest:
        raise BCActorIdentityError(
            "BC_BUNDLE_DIGEST_MISMATCH",
            f"checkpoint digest {recorded[:12]}… != bundle's "
            f"bc_parent_checkpoint_digest "
            f"{expected_checkpoint_digest[:12]}…")
    identity = payload.get("identity")
    if not isinstance(identity, Mapping):
        raise BCActorIdentityError("BC_IDENTITY_MISSING",
                                   "checkpoint carries no identity block")
    if str(identity.get("features_spec_digest")) != \
            expected_features_spec_digest:
        raise BCActorIdentityError(
            "BC_FEATURES_SPEC_MISMATCH",
            f"BC features spec digest "
            f"{str(identity.get('features_spec_digest'))[:12]}… != this "
            f"run's {expected_features_spec_digest[:12]}…")
    if str(identity.get("action_version")) != expected_action_version:
        raise BCActorIdentityError(
            "BC_ACTION_VERSION_MISMATCH",
            f"BC action version {identity.get('action_version')!r} != "
            f"{expected_action_version!r}")
    if int(identity.get("seed", -1)) != int(expected_seed):
        raise BCActorIdentityError(
            "BC_SEED_MISMATCH",
            f"BC seed {identity.get('seed')!r} != this run's seed "
            f"{expected_seed!r} (plan §I.11: same-seed BC actor)")
    model_state = payload.get("model_state")
    if not isinstance(model_state, Mapping) or \
            "encoder" not in model_state or "policy" not in model_state:
        raise BCActorIdentityError(
            "BC_STATE_MALFORMED",
            "model_state must carry the dual-TS policy encoder+head states")
    policy.load_state_dicts({"encoder": model_state["encoder"],
                             "policy": model_state["policy"]})
    return {
        "bc_trainer_version": str(payload.get("trainer_version", "")),
        "bc_checkpoint_digest": recorded,
        "features_spec_digest": str(identity.get("features_spec_digest")),
        "action_version": str(identity.get("action_version")),
        "seed": int(identity.get("seed")),
    }


# ---------------------------------------------------------------------------
# Registered polyak
# ---------------------------------------------------------------------------


class PolyakRegistry:
    """Registered (source, target) parameter pairs + one shared factor."""

    def __init__(self, torch, factor: float) -> None:
        if not (0.0 < factor < 1.0):
            raise DualTSIQLTrainerError(
                "POLYAK_INVALID", "target_polyak must be in (0, 1)")
        self.torch = torch
        self.factor = float(factor)
        self._pairs: list[tuple[str, tuple, tuple]] = []

    def register(self, name: str, source, target) -> None:
        src = [p for p in source.parameters()]
        dst = [p for p in target.parameters()]
        if len(src) != len(dst):
            raise DualTSIQLTrainerError(
                "POLYAK_SHAPE_MISMATCH", f"pair {name!r} parameter count")
        self._pairs.append((name, tuple(src), tuple(dst)))

    def update(self) -> None:
        with self.torch.no_grad():
            for _, src, dst in self._pairs:
                for p, pt in zip(src, dst):
                    pt.mul_(self.factor).add_(p, alpha=1.0 - self.factor)

    def payload(self) -> dict:
        return {"factor": self.factor,
                "pairs": [name for name, _, _ in self._pairs]}


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class DualTSIQLTrainer:
    """Phase-machined IQL over an :class:`EpisodeKeyedBuffer`."""

    def __init__(self, *, settings: DualTSIQLSettings,
                 buffer: EpisodeKeyedBuffer, run_dir: str | Path,
                 bc_checkpoint_path: str | Path, bundle: DualTSBundle,
                 features_spec_digest: str, s_g: float,
                 s_delta_e: float) -> None:
        if not (np.isfinite(s_g) and s_g > 0.0
                and np.isfinite(s_delta_e) and s_delta_e > 0.0):
            raise DualTSIQLTrainerError(
                "SCALE_INVALID", "s_g and s_delta_e must be finite > 0")
        self.settings = settings
        self.buffer = buffer
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.features_spec_digest = str(features_spec_digest)
        self.s_g = float(s_g)
        self.s_delta_e = float(s_delta_e)

        # -- identity gates on the bundle BEFORE anything else --------------
        bundle.validate()
        if bundle.schema_version != BUNDLE_SCHEMA_VERSION:
            raise DualTSIQLTrainerError(
                "BUNDLE_SCHEMA_UNKNOWN",
                f"expected {BUNDLE_SCHEMA_VERSION}")
        if bundle.action_version != ACTION_SCHEMA_VERSION:
            raise DualTSIQLTrainerError(
                "BUNDLE_ACTION_VERSION_MISMATCH",
                f"bundle action_version {bundle.action_version!r} != "
                f"{ACTION_SCHEMA_VERSION!r}")
        self.bundle = bundle
        self.bc_checkpoint_path = Path(bc_checkpoint_path)
        if not self.bc_checkpoint_path.is_file():
            raise DualTSIQLTrainerError(
                "BC_CHECKPOINT_MISSING", str(self.bc_checkpoint_path))

        self.torch, self.device = resolve_device(
            device=settings.device, require_cuda=settings.require_cuda)
        set_cpu_threads(intra=settings.cpu_intra_threads,
                        inter=settings.cpu_inter_threads)
        seed_everything(settings.seed)

        # -- BC actor (identity-gated load, then FROZEN for warmup) --------
        self.config = DualTSModelConfig(
            d_model=settings.d_model, d_element=settings.d_element,
            hidden=settings.hidden, n_layers=settings.n_layers,
            light=settings.light)
        self.actor = DualTSPolicy(
            config=self.config, action_max=settings.a_max_angstrom,
            sigma_range=(settings.sigma_min, 1.0),
            signed_mu_gate=settings.signed_mu_gate).to(self.device)
        bc_payload = self.torch.load(self.bc_checkpoint_path,
                                     map_location="cpu", weights_only=False)
        self.bc_provenance = load_bc_actor(
            self.torch, policy=self.actor, payload=bc_payload,
            expected_features_spec_digest=self.features_spec_digest,
            expected_action_version=ACTION_SCHEMA_VERSION,
            expected_seed=settings.seed,
            expected_checkpoint_digest=bundle.bc_parent_checkpoint_digest)
        self.bc_checkpoint_digest = self.bc_provenance["bc_checkpoint_digest"]
        self.freeze_actor()
        # frozen-actor reference: bit identity is verified at the phase
        # transition (belt-and-braces against any accidental actor write).
        self._actor_frozen_reference = {
            k: v.detach().to("cpu", copy=True)
            for k, v in self.actor.module.state_dict().items()}

        # -- independent Q1/Q2/V + registered polyak targets ---------------
        s_a = settings.s_a_angstrom  # None → a_max (S_A_RULE frozen)
        self.q1 = DualTSQCritic(config=self.config, s_a_angstrom=s_a,
                                a_max=settings.a_max_angstrom).to(self.device)
        self.q2 = DualTSQCritic(config=self.config, s_a_angstrom=s_a,
                                a_max=settings.a_max_angstrom).to(self.device)
        self.v = DualTSVCritic(config=self.config, s_a_angstrom=s_a,
                               a_max=settings.a_max_angstrom).to(self.device)
        self.q1_target = DualTSQCritic(
            config=self.config, s_a_angstrom=s_a,
            a_max=settings.a_max_angstrom).to(self.device)
        self.q2_target = DualTSQCritic(
            config=self.config, s_a_angstrom=s_a,
            a_max=settings.a_max_angstrom).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        for net in (self.q1_target, self.q2_target):
            for p in net.parameters():
                p.requires_grad_(False)
            net.train_mode(False)
        self.polyak = PolyakRegistry(self.torch, settings.target_polyak)
        for src_name, dst_name in POLYAK_PAIRS:
            self.polyak.register(src_name, getattr(self, src_name),
                                 getattr(self, dst_name))

        # -- optimizers -----------------------------------------------------
        self.opt_q = self.torch.optim.AdamW(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=settings.lr, weight_decay=settings.weight_decay)
        self.opt_v = self.torch.optim.AdamW(
            list(self.v.parameters()), lr=settings.lr,
            weight_decay=settings.weight_decay)
        self.opt_pi = self.torch.optim.AdamW(
            list(self.actor.parameters()), lr=settings.lr,
            weight_decay=settings.weight_decay)

        # -- per-head eligibility + §I.10 sampler + exposure accounting ------
        self.eligibility, self.eligibility_counts = \
            self._evaluate_head_eligibility()
        self.sampler = DualTSGroupBranchSampler(
            seed=settings.seed, groups=buffer.head_groups(),
            rows_per_group=settings.rows_per_group_per_draw)
        self.exposure = ExposureTracker(row_meta=[
            {
                "group_id": r.reaction_group_id,
                "state_id": r.transition.state_id,
                "branch_id": r.transition.branch_id,
                "remaining_arc_length": r.remaining_arc_length,
            }
            for r in buffer.rows
        ])
        self.skip_stats: dict[str, int] = {h: 0 for h in HEADS}

        # -- finding 5: near-TS integrity (remaining_arc_length) ------------
        self._ral_missing_count = self._check_ral_integrity()
        self._last_final_update = -1

        # -- phase machine ---------------------------------------------------
        self.phase = PHASE_WARMUP
        self.warmup_updates_done = 0
        self.joint_updates_done = 0
        self.phase_transitions: list[dict] = []
        self.history: list[dict] = []
        self._started = time.perf_counter()

    # -- freezing ------------------------------------------------------------

    def freeze_actor(self) -> None:
        for p in self.actor.parameters():
            p.requires_grad_(False)

    def unfreeze_actor(self) -> None:
        for p in self.actor.parameters():
            p.requires_grad_(True)

    def _verify_actor_frozen(self) -> None:
        for name, ref in self._actor_frozen_reference.items():
            live = self.actor.module.state_dict()[name].detach().cpu()
            if not self.torch.equal(live, ref):
                raise DualTSIQLTrainerError(
                    "ACTOR_CHANGED_DURING_WARMUP",
                    f"actor tensor {name!r} moved during warmup")

    def _enter_joint_phase(self) -> None:
        self._verify_actor_frozen()
        self.unfreeze_actor()
        self.phase = PHASE_JOINT
        self.phase_transitions.append({
            "event": "warmup->joint",
            "at_global_update": self.warmup_updates_done,
            "warmup_updates_done": self.warmup_updates_done,
            "actor_frozen_reference_verified": True,
        })

    # -- finding 5: near-TS integrity + exposure snapshots -------------------

    def _check_ral_integrity(self) -> int:
        """Formal runs refuse missing ``remaining_arc_length``; diagnostic
        runs warn and record the missing count (near-TS never silently 0)."""
        missing = [i for i, r in enumerate(self.buffer.rows)
                   if r.remaining_arc_length is None]
        if not missing:
            return 0
        if not self.settings.diagnostic_run:
            raise DualTSIQLTrainerError(
                "RAL_INTEGRITY_VIOLATION",
                f"{len(missing)} transition rows carry "
                f"remaining_arc_length=None in a formal (non-diagnostic) "
                f"IQL run; populating ral from the bound teacher table is a "
                f"WP6 prerequisite — refusing (pass diagnostic_run=True only "
                f"for research/fixture use)")
        print(f"[diagnostic] WARNING: {len(missing)} transition rows have "
              f"remaining_arc_length=None; near_ts_fraction will be reported "
              f"as None (never silently 0)", flush=True)
        return len(missing)

    def _snapshot_exposure(self, *, kind: str, update: int) -> None:
        epoch = self.sampler.draws // max(1, self.exposure.n_drawable)
        self.exposure.snapshot(kind=kind, update=int(update), epoch=epoch)

    def _snapshot_final(self) -> None:
        """Close a 'final' interval once per (update-count) boundary."""
        update = self.warmup_updates_done + self.joint_updates_done
        if update == self._last_final_update:
            return
        self._last_final_update = update
        self._snapshot_exposure(kind="final", update=update)

    # -- per-head eligibility (contract §3: 逐头资格) -------------------------

    def _evaluate_head_eligibility(self):
        eligibility: list[dict[str, bool]] = []
        counts = {h: {"indexed": 0, "skipped": 0, "skip_reasons": {}}
                  for h in HEADS}
        for idx, row in enumerate(self.buffer.rows):
            t = row.transition
            has_physical = (t.proposed_action.action_angstrom is not None)
            has_latent = (t.proposed_action.latent_u is not None)
            successor = self.buffer.next_row_of(idx)
            obs_ok = (successor is None
                      or t.next_observation_digest is not None)
            flags: dict[str, bool] = {}
            gates = {
                "Q": (has_physical and row.reward is not None
                      and not t.censored and obs_ok),
                "V": (has_physical and not t.censored),
                "actor": (has_latent and not t.censored),
            }
            for head in HEADS:
                entry = t.head_eligibility.get(head)
                if entry is None:
                    reason = "eligibility_not_declared"
                elif not bool(entry.get("eligible")):
                    reason = "flagged_ineligible"
                elif not gates[head]:
                    reason = {
                        "Q": ("reward_unknown" if row.reward is None else
                              "action_carrier_missing" if not has_physical
                              else "censored_not_q_eligible" if t.censored
                              else "successor_observation_missing"),
                        "V": ("action_carrier_missing" if not has_physical
                              else "censored_not_v_eligible"),
                        "actor": ("latent_carrier_missing"
                                  if not has_latent
                                  else "censored_not_actor_eligible"),
                    }[head]
                else:
                    reason = ""
                ok = reason == ""
                flags[head] = ok
                if ok:
                    counts[head]["indexed"] += 1
                else:
                    counts[head]["skipped"] += 1
                    counts[head]["skip_reasons"][reason] = \
                        counts[head]["skip_reasons"].get(reason, 0) + 1
            eligibility.append(flags)
        return eligibility, counts

    # -- batching -------------------------------------------------------------

    def _build_batch(self, rows: Sequence[DualTSTransitionRow],
                     row_indices: Sequence[int]) -> dict:
        torch = self.torch
        states = [r.state_inputs for r in rows]
        batch = collate_dual_states(torch, states)
        a_blocks = [torch.as_tensor(np.asarray(
            r.transition.proposed_action.action_angstrom,
            dtype=np.float64)) for r in rows]
        batch["a_angstrom"] = torch.cat(a_blocks, dim=0)
        n_atoms = [int(b.shape[0]) for b in a_blocks]
        batch_size = len(rows)
        batch["atom_sample_index"] = torch.repeat_interleave(
            torch.arange(batch_size), torch.as_tensor(n_atoms))
        batch["n_samples"] = batch_size
        # branch slots are PER-REACTION-GROUP branches (§I.10 hierarchy)
        branch_slot: dict[str, int] = {}
        branch_group: list[str] = []
        group_slot: dict[str, int] = {}
        sample_branch_index: list[int] = []
        for r in rows:
            key = f"{r.reaction_group_id}|{r.transition.branch_id}"
            if key not in branch_slot:
                branch_slot[key] = len(branch_slot)
                branch_group.append(r.reaction_group_id)
            if r.reaction_group_id not in group_slot:
                group_slot[r.reaction_group_id] = len(group_slot)
            sample_branch_index.append(branch_slot[key])
        batch["sample_branch_index"] = torch.as_tensor(
            np.asarray(sample_branch_index, dtype=np.int64))
        batch["n_branches"] = len(branch_slot)
        batch["branch_group_index"] = torch.as_tensor(np.asarray(
            [group_slot[g] for g in branch_group], dtype=np.int64))
        batch["n_groups"] = len(group_slot)
        # per-head masks + reward + bootstrap
        masks = {h: [] for h in HEADS}
        rewards: list[float] = []
        bootstrap: list[float] = []
        next_indices: list[int | None] = []
        for r, idx in zip(rows, row_indices):
            for h in HEADS:
                masks[h].append(1.0 if self.eligibility[idx][h] else 0.0)
            rewards.append(float(r.reward)
                           if r.reward is not None else 0.0)
            nxt = self.buffer.next_row_of(idx)
            next_indices.append(nxt)
            bootstrap.append(1.0 if nxt is not None else 0.0)
        for h in HEADS:
            batch[f"{h.lower()}_mask" if h != "actor" else "pi_mask"] = \
                torch.as_tensor(np.asarray(masks[h], dtype=np.float32))
        batch["reward"] = torch.as_tensor(np.asarray(rewards,
                                                     dtype=np.float32))
        batch["bootstrap_mask"] = torch.as_tensor(
            np.asarray(bootstrap, dtype=np.float32))
        # dataset latent for the actor log-prob (zero-filled where a row has
        # no latent carrier — such rows are excluded by pi_mask anyway)
        u_blocks = []
        for r in rows:
            u = r.transition.proposed_action.latent_u
            n = int(np.asarray(
                r.transition.proposed_action.action_angstrom).shape[0])
            if u is None:
                u_blocks.append(torch.zeros((n, 3), dtype=torch.float64))
            else:
                u_blocks.append(torch.as_tensor(np.asarray(u,
                                                           dtype=np.float64)))
        batch["expert_u"] = torch.cat(u_blocks, dim=0)
        batch = move_batch(batch, self.device)
        batch["_next_indices"] = next_indices
        batch["_row_indices"] = list(row_indices)
        return batch

    def _next_state_batch(self, batch: dict):
        """Collate the SAME-EPISODE successor states of bootstrapped rows."""
        pending = [(k, nxt) for k, nxt in enumerate(batch["_next_indices"])
                   if nxt is not None and float(batch["bootstrap_mask"][k]) > 0]
        if not pending:
            return None, pending
        states = [self.buffer.rows[nxt].state_inputs for _, nxt in pending]
        nb = collate_dual_states(self.torch, states)
        return move_batch(nb, self.device), pending

    # -- losses -----------------------------------------------------------------

    def _expectile_loss_masked(self, diff, mask):
        torch = self.torch
        tau = float(self.settings.expectile)
        w = torch.where(diff >= 0, torch.full_like(diff, tau),
                        torch.full_like(diff, 1.0 - tau))
        return (w * diff.pow(2) * mask).sum() / mask.sum().clamp(min=1.0)

    # -- one update ---------------------------------------------------------------

    def _one_update(self, *, policy_update: bool) -> dict:
        torch = self.torch
        row_indices = self.sampler.draw_batch(self.sampler.rows_per_round)
        self.exposure.record_batch(row_indices)
        rows = [self.buffer.rows[i] for i in row_indices]
        batch = self._build_batch(rows, row_indices)
        a = batch["a_angstrom"]
        ver = Q_ACTION_INPUT_VERSION
        entry: dict = {"phase": self.phase,
                       "global_update": self.warmup_updates_done
                       + self.joint_updates_done,
                       "rows_drawn": len(rows),
                       "actor_frozen": not policy_update}

        with torch.no_grad():
            qbar = torch.minimum(self.q1_target(batch, a, ver),
                                 self.q2_target(batch, a, ver))
        v_pred = self.v(batch)

        # ---- V: expectile-τ regression on the stop-grad target q̄ ---------
        v_mask = batch["v_mask"]
        loss_v = None
        if float(v_mask.sum()) > 0:
            loss_v = self._expectile_loss_masked(qbar - v_pred, v_mask)
            self.opt_v.zero_grad(set_to_none=True)
            loss_v.backward()
            torch.nn.utils.clip_grad_norm_(self.v.parameters(),
                                           self.settings.grad_clip)
            self.opt_v.step()
        else:
            self.skip_stats["V"] += 1

        # ---- Q: TD to r + γ·bootstrap·V(s_next), stop-grad ----------------
        with torch.no_grad():
            v_next_full = torch.zeros_like(qbar)
            next_batch, pending = self._next_state_batch(batch)
            if next_batch is not None:
                v_next = self.v(next_batch)
                for row_k, (_, _nxt) in enumerate(pending):
                    v_next_full[row_k] = v_next[row_k]
            y = batch["reward"].to(qbar.dtype) + \
                float(self.settings.gamma) * batch["bootstrap_mask"].to(
                    qbar.dtype) * v_next_full
        q_mask = batch["q_mask"]
        loss_q = None
        if float(q_mask.sum()) > 0:
            q1_pred = self.q1(batch, a, ver)
            q2_pred = self.q2(batch, a, ver)
            sq = 0.5 * ((q1_pred - y).pow(2) + (q2_pred - y).pow(2))
            loss_q = (sq * q_mask).sum() / q_mask.sum().clamp(min=1.0)
            self.opt_q.zero_grad(set_to_none=True)
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.q1.parameters()) + list(self.q2.parameters()),
                self.settings.grad_clip)
            self.opt_q.step()
        else:
            self.skip_stats["Q"] += 1

        # ---- registered polyak on the Q targets ----------------------------
        self.polyak.update()

        # ---- actor: AWR on the SAME §I.10 reduction as BC ------------------
        loss_pi = None
        if policy_update:
            pi_mask = batch["pi_mask"]
            if float(pi_mask.sum()) > 0:
                with torch.no_grad():
                    adv = qbar - v_pred.detach()
                    w = awr_weights(torch, adv,
                                    temperature=self.settings.advantage_temperature,
                                    clip=self.settings.weight_clip)
                out = self.actor(batch)
                lp = self.actor.log_prob_z(
                    out["mu"], out["log_sigma"], batch["expert_u"])
                weighted = lp["per_atom"] * w[batch["atom_sample_index"]]
                red, _ = hierarchical_logprob_reduction(
                    torch,
                    per_atom=weighted,
                    atom_sample_index=batch["atom_sample_index"],
                    n_samples=int(batch["n_samples"]),
                    sample_branch_index=batch["sample_branch_index"],
                    n_branches=int(batch["n_branches"]),
                    branch_group_index=batch["branch_group_index"],
                    n_groups=int(batch["n_groups"]))
                loss_pi = -red
                self.opt_pi.zero_grad(set_to_none=True)
                loss_pi.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.settings.grad_clip)
                self.opt_pi.step()
            else:
                self.skip_stats["actor"] += 1

        entry.update({
            "loss_v": None if loss_v is None else float(loss_v.detach()),
            "loss_q": None if loss_q is None else float(loss_q.detach()),
            "loss_pi": None if loss_pi is None else float(loss_pi.detach()),
            "q_finite": bool(torch.isfinite(qbar).all()
                             and torch.isfinite(v_pred).all()),
        })
        return entry

    # -- phase-machined loop ------------------------------------------------------

    def run_updates(self, n: int) -> list[dict]:
        entries: list[dict] = []
        for _ in range(int(n)):
            if self.phase == PHASE_WARMUP:
                entry = self._one_update(policy_update=False)
                self.warmup_updates_done += 1
                if self.warmup_updates_done >= self.settings.warmup_updates:
                    self._enter_joint_phase()
                    # finding 5: snapshot the exposure at the warmup→joint
                    # transition (history is populated for BOTH trainers).
                    self._snapshot_exposure(
                        kind="phase_transition",
                        update=self.warmup_updates_done)
            else:
                if self.joint_updates_done >= self.settings.joint_max_updates:
                    raise DualTSIQLTrainerError(
                        "JOINT_BUDGET_EXHAUSTED",
                        f"joint budget {self.settings.joint_max_updates} "
                        "updates exhausted (plan §I.11)")
                entry = self._one_update(policy_update=True)
                self.joint_updates_done += 1
            entries.append(entry)
            self.history.append(entry)
            # finding 5: snapshot every 1000 global updates (matches BC).
            global_update = self.warmup_updates_done + self.joint_updates_done
            if global_update % 1000 == 0:
                self._snapshot_exposure(kind="update_1000",
                                        update=global_update)
        return entries

    # -- TD inspection (fixture test (e)) ------------------------------------------

    def inspect_td_targets(self, row_indices: Sequence[int]) -> list[dict]:
        torch = self.torch
        rows = [self.buffer.rows[i] for i in row_indices]
        batch = self._build_batch(rows, row_indices)
        with torch.no_grad():
            v_next_full = torch.zeros(batch["reward"].shape[0],
                                      device=self.device)
            next_batch, pending = self._next_state_batch(batch)
            if next_batch is not None:
                v_next = self.v(next_batch)
                for k, _ in enumerate(pending):
                    v_next_full[k] = v_next[k]
            y = batch["reward"].to(v_next_full.dtype) + \
                float(self.settings.gamma) * batch["bootstrap_mask"].to(
                    v_next_full.dtype) * v_next_full
        out = []
        for k, idx in enumerate(row_indices):
            nxt = batch["_next_indices"][k]
            out.append({
                "transition_id": rows[k].transition.transition_id,
                "episode_id": rows[k].transition.episode_id,
                "successor_episode_id": (None if nxt is None else
                                         self.buffer.rows[nxt].transition.episode_id),
                "reward": float(batch["reward"][k]),
                "bootstrap_mask": float(batch["bootstrap_mask"][k]),
                "y": float(y[k]),
            })
        return out

    # -- identity -------------------------------------------------------------------

    def identity_payload(self) -> dict:
        return {
            "trainer_version": DUAL_TS_IQL_VERSION,
            "features_spec_digest": self.features_spec_digest,
            "action_version": ACTION_SCHEMA_VERSION,
            "action_input_version": Q_ACTION_INPUT_VERSION,
            "bundle_schema_version": BUNDLE_SCHEMA_VERSION,
            "bundle_digest": self.bundle.content_digest(),
            "bundle_run_id": self.bundle.run_id,
            "bc_checkpoint_digest": self.bc_checkpoint_digest,
            "bc_provenance": dict(self.bc_provenance),
            "frozen_scales": {
                "s_g": self.s_g,
                "s_delta_e": self.s_delta_e,
                "s_a_angstrom": float(
                    self.settings.s_a_angstrom
                    if self.settings.s_a_angstrom is not None
                    else self.settings.a_max_angstrom),
                "s_a_rule": S_A_RULE,
            },
            "targets": {
                "expectile": self.settings.expectile,
                "advantage_temperature": self.settings.advantage_temperature,
                "weight_clip": self.settings.weight_clip,
                "gamma": self.settings.gamma,
                "target_polyak": self.settings.target_polyak,
                "polyak_registry": self.polyak.payload(),
            },
            "sampling": {
                "rule": GROUP_BRANCH_SAMPLING_RULE,
                "within_branch_rule": WITHIN_BRANCH_RULE,
                "logprob_reduction_rule": LOGPROB_REDUCTION_RULE,
                "episode_chaining_rule":
                    EpisodeKeyedBuffer.EPISODE_CHAINING_RULE,
            },
            "critic_init_digests": {
                "q1": self.q1.init_state_digest,
                "q2": self.q2.init_state_digest,
                "v": self.v.init_state_digest,
            },
        }

    # -- checkpoint / resume -----------------------------------------------------------

    def _rng_payload(self) -> dict:
        torch = self.torch
        rng = {
            "torch_cpu": torch.get_rng_state(),
            "numpy": np.random.get_state(),
            "python": random.getstate(),
            "torch_cuda": [],
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            rng["torch_cuda"] = [torch.cuda.get_rng_state(i)
                                 for i in range(torch.cuda.device_count())]
        return rng

    def _restore_rng(self, rng: Mapping) -> None:
        torch = self.torch
        torch.set_rng_state(rng["torch_cpu"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
        cuda_states = list(rng.get("torch_cuda", []))
        if cuda_states and self.device.type == "cuda" \
                and torch.cuda.is_available():
            for i, state in enumerate(cuda_states):
                if i < torch.cuda.device_count():
                    torch.cuda.set_rng_state(state, i)

    def save_checkpoint(self, path: str | Path) -> str:
        torch = self.torch
        payload = {
            "schema_version": CHECKPOINT_SCHEMA,
            "trainer_version": DUAL_TS_IQL_VERSION,
            "phase": self.phase,
            "warmup_updates_done": int(self.warmup_updates_done),
            "joint_updates_done": int(self.joint_updates_done),
            "settings": self.settings.to_payload(),
            "identity": self.identity_payload(),
            "model_state": {
                "actor": {
                    block: {n: t.detach().to("cpu") for n, t in sd.items()}
                    for block, sd in self.actor.state_dicts().items()
                },
                "q1": {k: v.detach().to("cpu") for k, v in
                       self.q1.state_dict().items()},
                "q2": {k: v.detach().to("cpu") for k, v in
                       self.q2.state_dict().items()},
                "v": {k: v.detach().to("cpu") for k, v in
                      self.v.state_dict().items()},
                "q1_target": {k: v.detach().to("cpu") for k, v in
                              self.q1_target.state_dict().items()},
                "q2_target": {k: v.detach().to("cpu") for k, v in
                              self.q2_target.state_dict().items()},
            },
            "optimizer_state": {
                "q": self.opt_q.state_dict(),
                "v": self.opt_v.state_dict(),
                "pi": self.opt_pi.state_dict(),
            },
            "sampler_state": self.sampler.state_dict(),
            "rng": self._rng_payload(),
            "eligibility_counts": {
                h: dict(self.eligibility_counts[h]) for h in HEADS},
            "skip_stats": dict(self.skip_stats),
            "phase_transitions": list(self.phase_transitions),
            "history": list(self.history),
        }
        payload["content_digest"] = checkpoint_content_digest(torch, payload)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, target)
        return str(payload["content_digest"])

    def load_checkpoint_payload(self, payload: Mapping) -> None:
        torch = self.torch
        if payload.get("schema_version") != CHECKPOINT_SCHEMA:
            raise DualTSIQLTrainerError(
                "CHECKPOINT_SCHEMA_UNKNOWN",
                f"expected {CHECKPOINT_SCHEMA!r}")
        recorded = payload.get("content_digest")
        recomputed = checkpoint_content_digest(torch, payload)
        if not isinstance(recorded, str) or recorded != recomputed:
            raise DualTSIQLTrainerError(
                "CHECKPOINT_DIGEST_MISMATCH",
                f"recorded {str(recorded)[:12]}… != recomputed "
                f"{recomputed[:12]}… (tampered or corrupted checkpoint)")
        identity = payload["identity"]
        if str(identity["features_spec_digest"]) != \
                self.features_spec_digest or \
                str(identity["bc_checkpoint_digest"]) != \
                self.bc_checkpoint_digest or \
                str(identity["action_version"]) != ACTION_SCHEMA_VERSION:
            raise DualTSIQLTrainerError(
                "FROZEN_IDENTITY_MISMATCH",
                "checkpoint identity (spec/BC/action) does not match this "
                "run — refusing to resume across identities")
        frozen = identity["frozen_scales"]
        if (float(frozen["s_g"]) != self.s_g
                or float(frozen["s_delta_e"]) != self.s_delta_e
                or float(frozen["s_a_angstrom"]) != float(
                    self.settings.s_a_angstrom
                    if self.settings.s_a_angstrom is not None
                    else self.settings.a_max_angstrom)):
            raise DualTSIQLTrainerError(
                "FROZEN_SCALE_MISMATCH",
                "checkpoint frozen norm/scale params differ from this run")
        if dict(payload["settings"]) != self.settings.to_payload():
            raise DualTSIQLTrainerError(
                "SETTINGS_MISMATCH",
                "checkpoint settings differ from the trainer settings")
        # model states (strict)
        self.actor.load_state_dicts(dict(payload["model_state"]["actor"]))
        self.q1.load_state_dict(payload["model_state"]["q1"])
        self.q2.load_state_dict(payload["model_state"]["q2"])
        self.v.load_state_dict(payload["model_state"]["v"])
        self.q1_target.load_state_dict(payload["model_state"]["q1_target"])
        self.q2_target.load_state_dict(payload["model_state"]["q2_target"])
        self.opt_q.load_state_dict(payload["optimizer_state"]["q"])
        self.opt_v.load_state_dict(payload["optimizer_state"]["v"])
        self.opt_pi.load_state_dict(payload["optimizer_state"]["pi"])
        self.sampler.load_state_dict(payload["sampler_state"])
        self._restore_rng(payload["rng"])
        self.phase = str(payload["phase"])
        self.warmup_updates_done = int(payload["warmup_updates_done"])
        self.joint_updates_done = int(payload["joint_updates_done"])
        self.phase_transitions = list(payload["phase_transitions"])
        self.history = list(payload["history"])
        self.skip_stats = dict(payload["skip_stats"])
        if self.phase == PHASE_WARMUP:
            self.freeze_actor()
        else:
            self.unfreeze_actor()

    # -- reporting ------------------------------------------------------------------

    def run_record(self) -> dict:
        self._snapshot_final()
        exposure = self.exposure.payload()
        if self._ral_missing_count:
            # finding 5: never silently report near-TS 0 for missing ral.
            exposure = dict(exposure)
            exposure["near_ts_fraction"] = None
            exposure["near_ts_unknown_rows"] = int(self._ral_missing_count)
            exposure["near_ts_note"] = (
                "remaining_arc_length missing on some rows (diagnostic run): "
                "near-TS fraction reported as None, never silently 0")
        return {
            "schema_version": "pes2ts_training_run_v1",
            "trainer": "dual_ts_iql",
            "trainer_version": DUAL_TS_IQL_VERSION,
            "research_only": True,
            "phase": self.phase,
            "warmup_updates_done": self.warmup_updates_done,
            "joint_updates_done": self.joint_updates_done,
            "settings": self.settings.to_payload(),
            "identity": self.identity_payload(),
            "accounting": {
                "updates_run": self.warmup_updates_done
                + self.joint_updates_done,
                "batch_states": self.sampler.rows_per_round,
                "samples_exposed": self.sampler.draws,
            },
            "exposure": exposure,
            "eligibility_counts": self.eligibility_counts,
            "skip_stats": dict(self.skip_stats),
            "history": list(self.history),
            "seconds": round(time.perf_counter() - self._started, 2),
        }


def restore_dual_ts_iql_checkpoint(path: str | Path, *, buffer,
                                   bundle: DualTSBundle,
                                   bc_checkpoint_path: str | Path,
                                   features_spec_digest: str, s_g: float,
                                   s_delta_e: float, run_dir: str | Path,
                                   settings: DualTSIQLSettings | None = None,
                                   ) -> DualTSIQLTrainer:
    """Rebuild a trainer from a checkpoint and restore its FULL state.

    The checkpoint's own recorded settings are reused unless an explicit
    (identical) settings object is provided; frozen identity (features
    spec digest, BC digest, action version, scales) is verified against
    the caller's arguments inside the constructor and
    :meth:`load_checkpoint_payload`.
    """
    torch = require_torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise DualTSIQLTrainerError(
            "CHECKPOINT_SCHEMA_UNKNOWN", f"at {path}")
    if settings is None:
        settings = DualTSIQLSettings.from_payload(payload["settings"])
    trainer = DualTSIQLTrainer(
        settings=settings, buffer=buffer, run_dir=run_dir,
        bc_checkpoint_path=bc_checkpoint_path, bundle=bundle,
        features_spec_digest=features_spec_digest, s_g=s_g,
        s_delta_e=s_delta_e)
    trainer.load_checkpoint_payload(payload)
    return trainer


__all__ = [
    "BC_CHECKPOINT_SCHEMA",
    "BCActorIdentityError",
    "CHECKPOINT_SCHEMA",
    "DUAL_TS_IQL_VERSION",
    "DualTSTransitionRow",
    "DualTSIQLSettings",
    "DualTSIQLTrainer",
    "DualTSIQLTrainerError",
    "EpisodeKeyedBuffer",
    "PHASE_JOINT",
    "PHASE_WARMUP",
    "POLYAK_PAIRS",
    "PolyakRegistry",
    "awr_weights",
    "checkpoint_content_digest",
    "expectile_loss",
    "load_bc_actor",
    "restore_dual_ts_iql_checkpoint",
]
