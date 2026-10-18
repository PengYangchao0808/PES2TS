"""Offline conservative-Q + restricted-imitation control trainer (ADR-0017 §4).

Trains the Q head and the bounded residual head on REAL qualified
transitions collected under a frozen E_native environment.  Two explicit,
separable training modes (Demo24 plan §5/§7):

* ``cql`` — TD + conservative regularizer + restricted imitation.  The TD
  target ``y = r + max_a' Q_target(S', a')`` is computed ONLY when the
  transition's RECORDED ``bootstrap_mask`` is true (an ongoing decision with
  a genuinely recorded successor); the successor's own recorded context
  vector is used for ``S'`` — never the current decision's context, and
  never a fabricated zero-residual fallback.  Terminal and budget-truncated
  finals use ``y = r``: the status name alone never triggers bootstrapping.
* ``bc_only`` — restricted imitation only.  Setting ``cql_alpha=0`` is NOT
  sufficient to build this control group because TD would still update Q;
  the mode therefore freezes the Q head out of the optimizer and skips the
  TD/conservative/target-network machinery entirely.

* **conservative regularizer (CQL-style)** —
  ``alpha * (logsumexp_a Q(S, a) - Q(S, a_executed))`` over the recorded
  candidate set, pushing down Q on legal-but-unexecuted candidates so the
  policy does not extrapolate over the collection behaviour support.
* **restricted weighted imitation** — the residual head imitates only
  EXECUTED model/exploration commands with benefit support (accepted step
  with normalized cost within the per-step reference); rule-zero baseline
  actions and unexecuted counterfactuals never produce imitation labels.

Model/early-stopping selection uses ONLY reactions inside the train
selection, split into fit/dev BY REACTION GROUP (never by random frames);
the 8 valid cases never participate.  The random seeds are set BEFORE any
model is constructed.  The encoder stays frozen in the first round; weights
change only between rounds; nothing here runs online.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from pes2ts_core.pesn.contracts.control import CONTROL_CONTEXT_DIM
from pes2ts_core.pesn.models.control_heads import ControlNet
from pes2ts_core.pesn.offline.experiments.runs import RunRecord
from pes2ts_core.pesn.offline.transitions import (
    QualifiedTransition,
    context_vector_from_doc,
)

CONTROL_RUN_PREFIX = "control"
BEST_CHECKPOINT_NAME = "checkpoint_best.pt"
LAST_CHECKPOINT_NAME = "checkpoint_last.pt"
TRAINING_MODES = ("cql", "bc_only")
_Q_PARAM_PREFIXES = ("q_driver", "q_final")


def _tensor_digest(tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def _module_param_digests(module) -> dict[str, dict[str, str]]:
    """{param_name: {sha256, shape}} for digest-of-weights evidence."""
    digests: dict[str, dict[str, str]] = {}
    for name, parameter in module.named_parameters():
        digests[name] = {
            "sha256": _tensor_digest(parameter),
            "shape": json.dumps(list(parameter.shape)),
        }
    return digests


def _digest_map_sha(digests: dict[str, dict[str, str]]) -> str:
    return hashlib.sha256(
        json.dumps(digests, sort_keys=True).encode("utf-8")
    ).hexdigest()


class ControlTrainerError(RuntimeError):
    """Typed control-trainer failure."""


@dataclass(slots=True)
class ControlTrainSettings:
    batch_transitions: int = 16
    max_epochs: int = 60
    patience: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-4
    clip: float = 1.0
    cql_alpha: float = 1.0
    imitation_weight: float = 0.5
    target_sync_epochs: int = 4
    freeze_encoder: bool = True
    encoder_init_bundle: str | None = None
    training_mode: str = "cql"
    dev_fraction: float = 0.25
    seed: int = 42
    hidden: int = 64
    n_layers: int = 3
    n_basis: int = 16
    use_coordinate_attention: bool = False
    step_cost_reference_seconds: float = 30.0

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "ControlTrainSettings":
        train_cfg = config.get("control_train", {})
        model_cfg = config.get("model", {})
        reward_cfg = config.get("reward", {})
        settings = cls(
            batch_transitions=int(train_cfg.get("batch_transitions", 16)),
            max_epochs=int(train_cfg.get("max_epochs", 60)),
            patience=int(train_cfg.get("patience", 8)),
            lr=float(train_cfg.get("lr", 3e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
            clip=float(train_cfg.get("clip", 1.0)),
            cql_alpha=float(train_cfg.get("cql_alpha", 1.0)),
            imitation_weight=float(train_cfg.get("imitation_weight", 0.5)),
            target_sync_epochs=int(train_cfg.get("target_sync_epochs", 4)),
            freeze_encoder=bool(train_cfg.get("freeze_encoder", True)),
            encoder_init_bundle=train_cfg.get("encoder_init_bundle"),
            training_mode=str(train_cfg.get("training_mode", "cql")),
            dev_fraction=float(train_cfg.get("dev_fraction", 0.25)),
            seed=int(config.get("seed", 42)),
            hidden=int(model_cfg.get("hidden", 64)),
            n_layers=int(model_cfg.get("n_layers", 3)),
            n_basis=int(model_cfg.get("n_basis", 16)),
            use_coordinate_attention=bool(
                model_cfg.get("use_coordinate_attention", False)
            ),
            step_cost_reference_seconds=float(
                reward_cfg.get("step_cost_reference_seconds", 30.0)
            ),
        )
        if settings.training_mode not in TRAINING_MODES:
            raise ControlTrainerError(
                f"training_mode {settings.training_mode!r} not in {list(TRAINING_MODES)}"
            )
        if not 0.0 <= settings.dev_fraction < 1.0:
            raise ControlTrainerError("dev_fraction must be in [0, 1)")
        return settings


def build_control_model(settings: ControlTrainSettings) -> ControlNet:
    return ControlNet.build(
        hidden=settings.hidden,
        n_layers=settings.n_layers,
        n_basis=settings.n_basis,
        context_dim=CONTROL_CONTEXT_DIM,
        use_attention=settings.use_coordinate_attention,
    )


def split_fit_dev_reactions(
    transitions: Sequence[QualifiedTransition],
    *,
    dev_fraction: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Deterministic reaction-group split: whole reactions, never frames."""
    reactions = sorted({t.reaction_id for t in transitions})
    if len(reactions) < 2 or dev_fraction <= 0.0:
        return reactions, []
    rng = np.random.default_rng(seed)
    order = list(reactions)
    rng.shuffle(order)
    n_dev = max(1, min(len(order) - 1, int(round(dev_fraction * len(order)))))
    return sorted(order[n_dev:]), sorted(order[:n_dev])


@dataclass(slots=True)
class ControlBatch:
    """One training batch of qualified transitions.

    ``next_contexts`` carries the recorded context vector of each
    transition's SUCCESSOR decision (None when no bootstrap is legal).
    """

    transitions: list[QualifiedTransition]
    next_contexts: list[tuple[float, ...] | None]


class ControlTrainer:
    """One offline control-training attempt bound to a run record."""

    def __init__(
        self,
        *,
        settings: ControlTrainSettings,
        run_record: RunRecord | None,
        config_sha256: str,
        transitions_digest: str,
        support_registry: Any | None = None,
    ) -> None:
        self.settings = settings
        self.run_record = run_record
        self.config_sha256 = config_sha256
        self.transitions_digest = transitions_digest
        self.support_registry = support_registry
        self.bootstrap_gate_mode = (
            "registry_v1" if support_registry is not None
            else "legacy_recorded_candidates"
        )
        self.n_bootstrap_cancelled_no_support = 0
        self.bootstrap_unsupported_reasons: dict[str, int] = {}
        # Seeds are set BEFORE any model tensor is created (Demo24 plan §5).
        np.random.seed(settings.seed)
        import torch as _torch

        _torch.manual_seed(settings.seed)
        self.model = build_control_model(settings)
        self.torch = _torch
        if settings.encoder_init_bundle:
            self._init_encoder_from_bundle(settings.encoder_init_bundle)
        self.encoder_params_sha256_before = _digest_map_sha(
            _module_param_digests(self.model.encoder.module)
        )
        self.q_params_before = {
            name: digest
            for name, digest in _module_param_digests(
                self.model.heads.module
            ).items()
            if name.startswith(_Q_PARAM_PREFIXES)
        }
        self.q_params_sha256_before = _digest_map_sha(self.q_params_before)
        if settings.freeze_encoder:
            frozen = self.model.freeze_encoder()
            self.n_encoder_frozen_tensors = frozen
            if self.run_record:
                self.run_record.append_event("encoder_frozen", n_tensors=frozen)
        else:
            self.n_encoder_frozen_tensors = 0
        if settings.training_mode == "bc_only":
            n_q_frozen = self._freeze_q_head(self.model)
            if self.run_record:
                self.run_record.append_event(
                    "q_head_frozen",
                    reason="bc_only_training_mode",
                    n_tensors=n_q_frozen,
                )
        self.optimizer = _torch.optim.AdamW(
            self.model.trainable_parameters(),
            lr=settings.lr,
            weight_decay=settings.weight_decay,
        )
        optimizer_param_names = [
            name
            for name, parameter in self.model.heads.module.named_parameters()
            if parameter.requires_grad
        ]
        optimizer_param_names += [
            f"encoder.{name}"
            for name, parameter in self.model.encoder.module.named_parameters()
            if parameter.requires_grad
        ]
        self.optimizer_param_names_sha256 = hashlib.sha256(
            json.dumps(sorted(optimizer_param_names)).encode("utf-8")
        ).hexdigest()
        self.n_parameters_total = self.model.n_parameters()
        self.n_parameters_frozen = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if not parameter.requires_grad
        )
        self.n_parameters_trainable = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )
        if settings.training_mode == "bc_only":
            self.target_model = None
        else:
            self.target_model = build_control_model(settings)
            self.target_model.load_state_dicts(self.model.state_dicts())
            for parameter in self.target_model.parameters():
                parameter.requires_grad_(False)
            self.target_model.train_mode(False)
        self.best_dev_loss = float("inf")
        self.best_epoch = -1
        self._epochs_without_improvement = 0
        self.n_bootstrap_inconsistent = 0
        trainable = self.model.trainable_parameters()
        self._grad_anchor = trainable[0] if trainable else None
        if self.run_record:
            self.run_record.update_status(
                "running",
                n_parameters=self.n_parameters_total,
                n_parameters_total=self.n_parameters_total,
                n_parameters_frozen=self.n_parameters_frozen,
                n_parameters_trainable=self.n_parameters_trainable,
                training_mode=settings.training_mode,
                optimizer_param_names_sha256=self.optimizer_param_names_sha256,
                q_params_sha256_before=self.q_params_sha256_before,
                encoder_params_sha256_before=self.encoder_params_sha256_before,
            )

    @staticmethod
    def _freeze_q_head(model) -> int:
        """Structurally freeze the Q-only modules (q_driver, q_final)."""
        frozen = 0
        for name, parameter in model.heads.module.named_parameters():
            if name.startswith("q_driver") or name.startswith("q_final"):
                parameter.requires_grad_(False)
                frozen += 1
        return frozen

    def _trainable_parameters(self):
        """bc_only trains the residual head only — Q can never drift via TD."""
    def _init_encoder_from_bundle(self, bundle_dir: str) -> None:
        from pes2ts_core.pesn.bundle.reader import load_bundle_weights

        states = load_bundle_weights(bundle_dir)
        model_state = states.get("model_state", states)
        if "encoder" in model_state:
            self.model.load_encoder_state(model_state["encoder"])
        else:
            raise ControlTrainerError(
                f"bundle {bundle_dir} does not carry an encoder state to reuse"
            )

    # -- feature building ---------------------------------------------------
    def _features_for(self, transition: QualifiedTransition):
        from pes2ts_core.pesn.features.graph import build_features

        return build_features(transition.state_input())

    def _next_features_for(self, transition: QualifiedTransition):
        from pes2ts_core.generation.planning.policy_adapter import (
            coordinate_spec_from_payload,
            task_graph_from_payload,
        )
        from pes2ts_core.pesn.contracts.inputs import build_pesn_input
        from pes2ts_core.pesn.features.graph import build_features

        sidecar = transition.case
        base = build_pesn_input(
            input_id=f"{transition.reaction_id}#next{transition.doc['decision_index']}",
            task_graph=task_graph_from_payload(sidecar["task_graph"]),
            coordinate_spec=coordinate_spec_from_payload(sidecar["coordinate_spec"]),
            coordinates=transition.doc["state_after_geometry"],
            charge=sidecar.get("charge"),
            multiplicity=sidecar.get("multiplicity"),
            method="gfn2_xtb_e_native",
        )
        return build_features(base)

    def _successor_support_mask(
        self,
        transition: QualifiedTransition,
        next_context: tuple[float, ...],
        next_candidates: Sequence[Sequence[float]],
    ) -> list[bool]:
        """Post-gate support of the successor's recorded candidate actions.

        Without a registry this is the legacy behaviour (recorded candidates
        only); with one, the TD max sees exactly the same mask semantics the
        execution-side Q gate uses (plan §5.2).  The old transition's
        ``bootstrap_mask`` is never rewritten — cancellation is recorded in
        the run's own counters.
        """
        if self.support_registry is None:
            return [True] * len(next_candidates)
        from pes2ts_core.pesn.offline.action_support import coordinate_spec_sha256

        verdicts = self.support_registry.query_actions(
            reaction_id=transition.reaction_id,
            driver_ids=transition.doc["action"]["driver_ids"],
            epsilon=transition.doc["action"]["epsilon"],
            environment_hash=str(transition.doc["environment_hash"]),
            reward_spec_sha256=self.support_registry.reward_spec_sha256,
            coordinate_spec_sha256=coordinate_spec_sha256(transition.case),
            geometry=transition.doc["state_after_geometry"],
            context_vector=list(next_context),
            candidate_actions=[list(c) for c in next_candidates],
        )
        for verdict in verdicts:
            if not verdict.supported and verdict.reason:
                self.bootstrap_unsupported_reasons[verdict.reason] = (
                    self.bootstrap_unsupported_reasons.get(verdict.reason, 0) + 1
                )
        return [verdict.supported for verdict in verdicts]

    # -- losses -------------------------------------------------------------
    def _batch_tensors(self, batch: ControlBatch):
        torch = self.torch
        features = [self._features_for(t) for t in batch.transitions]
        contexts = torch.as_tensor(
            np.asarray(
                [context_vector_from_doc(t.doc) for t in batch.transitions],
                dtype=np.float32,
            )
        )
        return features, contexts

    def _score_state_candidates(self, model, features, contexts, candidates_per_state):
        """Flat (candidate, driver) Q scoring across a batch of states."""
        torch = self.torch
        rows: list[float] = []
        slots: list[int] = []
        driver_rows: list[int] = []
        slot_counter = 0
        driver_starts: list[int] = []
        position = 0
        for feature in features:
            driver_starts.append(position)
            position += len(feature.driver_edge_index)
        for state_index, candidates in enumerate(candidates_per_state):
            base = driver_starts[state_index]
            for candidate in candidates:
                for driver_row, u_j in enumerate(candidate):
                    rows.append(float(u_j))
                    slots.append(slot_counter)
                    driver_rows.append(base + driver_row)
                slot_counter += 1
        n_candidates = slot_counter
        if not rows:
            return torch.zeros((n_candidates,)), []
        q = model.forward_q(
            features,
            contexts,
            torch.as_tensor(rows, dtype=torch.float32),
            torch.as_tensor(slots, dtype=torch.long),
            torch.as_tensor(driver_rows, dtype=torch.long),
            n_candidates,
        )
        offsets: list[int] = []
        cursor = 0
        for candidates in candidates_per_state:
            offsets.append(cursor)
            cursor += len(candidates)
        return q, offsets

    def compute_losses(self, batch: ControlBatch):
        """TD + conservative + restricted-imitation losses for one batch."""
        torch = self.torch
        transitions = batch.transitions
        if not transitions:
            raise ControlTrainerError("empty batch")
        features, contexts = self._batch_tensors(batch)
        candidates_per_state = [t.candidates for t in transitions]
        executed_indices = [t.executed_index for t in transitions]
        mode = self.settings.training_mode

        zero = None
        if mode == "cql":
            q_flat, offsets = self._score_state_candidates(
                self.model, features, contexts, candidates_per_state
            )
            executed_q = torch.stack(
                [
                    q_flat[offsets[state] + executed_indices[state]]
                    for state in range(len(transitions))
                ]
            )
        else:
            q_flat = None
            offsets = []
            executed_q = None
            zero = (
                self._grad_anchor.sum() * 0.0
                if self._grad_anchor is not None
                else contexts.sum() * 0.0
            )

        td_loss = zero if zero is not None else executed_q.sum() * 0.0
        conservative = zero if zero is not None else executed_q.sum() * 0.0
        if mode == "cql":
            # TD targets from the frozen target network, gated by the
            # RECORDED bootstrap_mask; the successor's own recorded context
            # vector forms S' (never the current decision's context, never a
            # fabricated fallback candidate set).
            with torch.no_grad():
                next_q_list: list[float] = []
                bootstrap_active: list[bool] = []
                for state, transition in enumerate(transitions):
                    next_candidates = transition.next_candidates
                    next_context = batch.next_contexts[state]
                    if not (transition.bootstrap_mask and next_candidates
                            and next_context is not None):
                        if transition.bootstrap_mask:
                            self.n_bootstrap_inconsistent += 1
                        next_q_list.append(0.0)
                        bootstrap_active.append(False)
                        continue
                    next_features = [self._next_features_for(transition)]
                    next_context_tensor = torch.as_tensor(
                        np.asarray(next_context, dtype=np.float32)
                    ).reshape(1, self.model.heads.context_dim)
                    next_q, _unused = self._score_state_candidates(
                        self.target_model,
                        next_features,
                        next_context_tensor,
                        [next_candidates],
                    )
                    supported_mask = self._successor_support_mask(
                        transition, next_context, next_candidates
                    )
                    if not any(supported_mask):
                        self.n_bootstrap_cancelled_no_support += 1
                        next_q_list.append(0.0)
                        bootstrap_active.append(False)
                        continue
                    values = [
                        float(next_q[index].item())
                        for index, supported in enumerate(supported_mask)
                        if supported
                    ]
                    next_q_list.append(max(values))
                    bootstrap_active.append(True)
                rewards = torch.as_tensor(
                    [t.reward for t in transitions], dtype=torch.float32
                )
                bootstrap = torch.as_tensor(
                    [1.0 if active else 0.0 for active in bootstrap_active],
                    dtype=torch.float32,
                )
                targets = rewards + bootstrap * torch.as_tensor(
                    next_q_list, dtype=torch.float32
                )
            td_loss = torch.nn.functional.smooth_l1_loss(executed_q, targets)

            if q_flat is not None and q_flat.numel() > 1:
                grouped = torch.stack(
                    [
                        torch.logsumexp(
                            q_flat[offsets[state]: offsets[state] + len(cands)], dim=0
                        )
                        for state, cands in enumerate(candidates_per_state)
                    ]
                )
                conservative = (grouped - executed_q).mean()
            else:
                conservative = q_flat.sum() * 0.0

        imitation = zero if zero is not None else executed_q.sum() * 0.0
        n_imitation = 0
        imitation_states: list[int] = []
        imitation_targets: list[tuple[float, ...]] = []
        for state, transition in enumerate(transitions):
            if _has_benefit_support(transition, self.settings):
                imitation_states.append(state)
                imitation_targets.append(
                    transition.candidates[transition.executed_index]
                )
        if imitation_states and self.settings.imitation_weight > 0:
            residual_outputs = self.model.forward_residual(
                [features[state] for state in imitation_states],
                contexts[imitation_states],
            )
            u_flat = residual_outputs["residual_components"]
            counts = [
                int(c)
                for c in residual_outputs["driver_counts"].cpu().numpy()
            ]
            predictions: list[Any] = []
            cursor = 0
            for position in range(len(imitation_states)):
                count = counts[position]
                predictions.append(u_flat[cursor: cursor + count])
                cursor += count
            target_tensors = [
                torch.as_tensor(list(target), dtype=torch.float32)
                for target in imitation_targets
            ]
            squared = [
                ((pred - target) ** 2).mean()
                for pred, target in zip(predictions, target_tensors)
            ]
            imitation = torch.stack(squared).mean()
            n_imitation = len(imitation_states)

        if mode == "bc_only":
            total = self.settings.imitation_weight * imitation
        else:
            total = (
                td_loss
                + self.settings.cql_alpha * conservative
                + self.settings.imitation_weight * imitation
            )
        return {
            "total": total,
            "td": td_loss,
            "conservative": conservative,
            "imitation": imitation,
            "n_imitation": n_imitation,
            "imitation_states": tuple(imitation_states),
        }

    # -- loop ---------------------------------------------------------------
    def train(self, transitions: Sequence[QualifiedTransition]) -> dict[str, Any]:
        if not transitions:
            raise ControlTrainerError("no qualified transitions to train on")
        fit_reactions, dev_reactions = split_fit_dev_reactions(
            transitions,
            dev_fraction=self.settings.dev_fraction,
            seed=self.settings.seed,
        )
        fit_set = set(fit_reactions)
        dev_set = set(dev_reactions)
        fit_transitions = [t for t in transitions if t.reaction_id in fit_set]
        dev_transitions = [t for t in transitions if t.reaction_id in dev_set]
        uses_dev = bool(dev_transitions)

        successor_contexts = _successor_context_index(transitions)

        def with_next_contexts(chunk: list[QualifiedTransition]) -> ControlBatch:
            return ControlBatch(
                transitions=chunk,
                next_contexts=[
                    successor_contexts.get(
                        (
                            t.episode_successor_key_base,
                            int(t.doc["decision_index"]) + 1,
                        )
                    )
                    for t in chunk
                ],
            )

        rng = np.random.default_rng(self.settings.seed)
        indices = np.arange(len(fit_transitions))
        n_imitation_total = 0
        imitation_distinct_ids: set[str] = set()
        if self.run_record:
            self.run_record.append_event(
                "control_train_start",
                n_transitions=len(transitions),
                training_mode=self.settings.training_mode,
                n_fit_transitions=len(fit_transitions),
                n_dev_transitions=len(dev_transitions),
                fit_reactions=fit_reactions,
                dev_reactions=dev_reactions,
                n_parameters_total=self.n_parameters_total,
                n_parameters_frozen=self.n_parameters_frozen,
                n_parameters_trainable=self.n_parameters_trainable,
                optimizer_param_names_sha256=self.optimizer_param_names_sha256,
            )
        for epoch in range(self.settings.max_epochs):
            started = time.perf_counter()
            self.model.train_mode(True)
            rng.shuffle(indices)
            sums = {"total": 0.0, "td": 0.0, "conservative": 0.0, "imitation": 0.0}
            n_imitation = 0
            n_batches = 0
            for start in range(0, len(indices), self.settings.batch_transitions):
                chunk = [
                    fit_transitions[int(i)]
                    for i in indices[start: start + self.settings.batch_transitions]
                ]
                batch = with_next_contexts(chunk)
                self.optimizer.zero_grad()
                losses = self.compute_losses(batch)
                losses["total"].backward()
                self.torch.nn.utils.clip_grad_norm_(
                    self.model.trainable_parameters(), self.settings.clip
                )
                self.optimizer.step()
                n_batches += 1
                for key in sums:
                    sums[key] += float(losses[key].item())
                n_imitation += int(losses["n_imitation"])
                imitation_distinct_ids.update(
                    str(chunk[state].doc.get("transition_id"))
                    for state in losses.get("imitation_states", ())
                )
            epoch_loss = sums["total"] / max(1, n_batches)
            n_imitation_total += n_imitation
            if uses_dev:
                selection_loss = self._evaluate_loss(
                    with_next_contexts(dev_transitions)
                )
            else:
                selection_loss = epoch_loss
            if self.run_record:
                self.run_record.add_epoch(
                    {
                        "epoch": epoch,
                        "seconds": round(time.perf_counter() - started, 2),
                        "loss_total": round(epoch_loss, 6),
                        "loss_td": round(sums["td"] / max(1, n_batches), 6),
                        "loss_conservative": round(
                            sums["conservative"] / max(1, n_batches), 6
                        ),
                        "loss_imitation": round(
                            sums["imitation"] / max(1, n_batches), 6
                        ),
                        "selection_loss": round(float(selection_loss), 6),
                        "selection_basis": "dev_reactions" if uses_dev else "train_fallback",
                        "n_imitation": n_imitation,
                    }
                )
            if selection_loss < self.best_dev_loss:
                self.best_dev_loss = float(selection_loss)
                self.best_epoch = epoch
                self._epochs_without_improvement = 0
                self._save_checkpoint(BEST_CHECKPOINT_NAME)
            else:
                self._epochs_without_improvement += 1
            self._save_checkpoint(LAST_CHECKPOINT_NAME)
            if (
                self.settings.training_mode == "cql"
                and (epoch + 1) % max(1, self.settings.target_sync_epochs) == 0
            ):
                self.target_model.load_state_dicts(self.model.state_dicts())
                for parameter in self.target_model.parameters():
                    parameter.requires_grad_(False)
                self.target_model.train_mode(False)
            if self._epochs_without_improvement >= self.settings.patience:
                if self.run_record:
                    self.run_record.append_event(
                        "early_stop", epoch=epoch, best_epoch=self.best_epoch
                    )
                break
        q_params_after = {
            name: digest
            for name, digest in _module_param_digests(
                self.model.heads.module
            ).items()
            if name.startswith(_Q_PARAM_PREFIXES)
        }
        q_params_sha256_after = _digest_map_sha(q_params_after)
        encoder_params_sha256_after = _digest_map_sha(
            _module_param_digests(self.model.encoder.module)
        )
        q_params_changed = q_params_sha256_after != self.q_params_sha256_before
        encoder_params_changed = (
            encoder_params_sha256_after != self.encoder_params_sha256_before
        )
        if self.settings.training_mode == "bc_only" and q_params_changed:
            raise ControlTrainerError(
                "bc_only training changed Q-specific parameters "
                f"({self.q_params_sha256_before[:12]} -> {q_params_sha256_after[:12]}); "
                "the BC control group is invalid"
            )
        if self.settings.freeze_encoder and encoder_params_changed:
            raise ControlTrainerError(
                "frozen encoder parameters changed during training "
                f"({self.encoder_params_sha256_before[:12]} -> "
                f"{encoder_params_sha256_after[:12]})"
            )
        if self.run_record:
            self.run_record.append_event(
                "control_train_end",
                training_mode=self.settings.training_mode,
                q_params_sha256_after=q_params_sha256_after,
                encoder_params_sha256_after=encoder_params_sha256_after,
                q_params_changed=q_params_changed,
                encoder_params_changed=encoder_params_changed,
                n_imitation_distinct_samples=len(imitation_distinct_ids),
                n_imitation_epoch_uses=n_imitation_total,
            )
        return {
            "training_mode": self.settings.training_mode,
            "best_epoch": self.best_epoch,
            "best_loss": (
                None if np.isinf(self.best_dev_loss) else round(self.best_dev_loss, 6)
            ),
            "selection_basis": "dev_reactions" if uses_dev else "train_fallback",
            "fit_reactions": fit_reactions,
            "dev_reactions": dev_reactions,
            "n_bootstrap_inconsistent": self.n_bootstrap_inconsistent,
            "bootstrap_support": {
                "gate": self.bootstrap_gate_mode,
                "n_cancelled_no_supported_successor_action": (
                    self.n_bootstrap_cancelled_no_support
                ),
                "unsupported_reasons": dict(self.bootstrap_unsupported_reasons),
            },
            "n_imitation_total": n_imitation_total,
            "n_imitation_distinct_samples": len(imitation_distinct_ids),
            "n_imitation_epoch_uses": n_imitation_total,
            "n_parameters": self.model.n_parameters(),
            "n_parameters_total": self.n_parameters_total,
            "n_parameters_frozen": self.n_parameters_frozen,
            "n_parameters_trainable": self.n_parameters_trainable,
            "optimizer_param_names_sha256": self.optimizer_param_names_sha256,
            "q_params_sha256_before": self.q_params_sha256_before,
            "q_params_sha256_after": q_params_sha256_after,
            "q_params_changed": q_params_changed,
            "encoder_params_sha256_before": self.encoder_params_sha256_before,
            "encoder_params_sha256_after": encoder_params_sha256_after,
            "encoder_params_changed": encoder_params_changed,
            "encoder_frozen": self.settings.freeze_encoder,
        }

    def _evaluate_loss(self, batch: ControlBatch) -> float:
        """Loss on the selection (dev) split — no gradients, no stepping."""
        self.model.train_mode(False)
        with self.torch.no_grad():
            losses = self.compute_losses(batch)
        return float(losses["total"].item())

    def _save_checkpoint(self, name: str) -> None:
        if not self.run_record:
            return
        torch = self.torch
        payload = {
            "schema_version": "pesn_control_checkpoint_v1",
            "model_state": self.model.state_dicts(),
            "config_sha256": self.config_sha256,
            "transitions_digest": self.transitions_digest,
            "training_mode": self.settings.training_mode,
        }
        torch.save(payload, self.run_record.run_dir / name)


def _successor_context_index(
    transitions: Sequence[QualifiedTransition],
) -> dict[tuple[str, int], tuple[float, ...]]:
    """(episode-or-reaction key, decision_index) -> recorded context.

    Episode-aware successor identity: transitions carrying episode-index
    metadata register and link strictly inside their own episode, and only
    when the recorded geometric link (state_after == next decision_start)
    is compatible.  Legacy transitions without episode metadata keep the
    historical same-reaction linking; equal (reaction, decision_index)
    pairs from different episodes can never chain.
    """
    index: dict[tuple[str, int], tuple[float, ...]] = {}
    for transition in transitions:
        if transition.episode is not None and not (
            transition.incoming_link_compatible
        ):
            continue
        key = (
            transition.episode_successor_key_base,
            int(transition.doc["decision_index"]),
        )
        try:
            index[key] = context_vector_from_doc(transition.doc)
        except Exception:  # noqa: BLE001 - defensive: malformed context rows
            continue
    return index


def _has_benefit_support(
    transition: QualifiedTransition, settings: ControlTrainSettings
) -> bool:
    """Executed model/exploration command, accepted, normalized cost in range."""
    action = transition.doc["action"]
    if str(action["source"]) == "rule_zero":
        return False
    if not bool(transition.doc.get("accepted")):
        return False
    cost = float(transition.doc["macro_cost"]["wall_seconds"])
    return cost <= settings.step_cost_reference_seconds


__all__ = [
    "BEST_CHECKPOINT_NAME",
    "ControlBatch",
    "ControlTrainer",
    "ControlTrainerError",
    "ControlTrainSettings",
    "LAST_CHECKPOINT_NAME",
    "TRAINING_MODES",
    "build_control_model",
    "split_fit_dev_reactions",
]
