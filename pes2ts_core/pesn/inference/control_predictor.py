"""Runtime control predictor: bounded residual + candidate Q_proxy (ADR-0017).

``ControlPredictor.from_bundle`` loads a validated ``pesn_bundle_format_v2``
research bundle and serves per-state residual proposals plus per-candidate
Q_proxy estimates.  This module is pure inference: it imports nothing from
``generation/``, ``integration/``, or ``offline/`` (product boundary), never
constructs commands, and never touches the filesystem beyond the bundle.

Both heads are research-only.  An unavailable residual head returns a typed
reason (never silent zeros); Q estimates are parallel to the caller's frozen
candidate list, and candidates without training support carry
``no_action_support``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.bundle.reader import (
    BundleError,
    SUPPORTED_BUNDLE_FORMAT_V2,
    load_bundle_weights,
    read_bundle_manifest,
)
from pes2ts_core.pesn.contracts.control import (
    CAP_Q_PROXY,
    CAP_RESIDUAL,
    CONTROL_CONTEXT_DIM,
    CONTROL_INPUT_SCHEMA_VERSION,
    CONTROL_PREDICTION_SCHEMA_VERSION,
    CandidateQEstimate,
    ControlInput,
    ControlPrediction,
)
from pes2ts_core.pesn.features.graph import build_features

CONTROL_PREDICTION_SCHEMA: Final[str] = CONTROL_PREDICTION_SCHEMA_VERSION
_CONTROL_INPUT_SCHEMA: Final[str] = CONTROL_INPUT_SCHEMA_VERSION


class ControlPredictor:
    """Loaded research control model with per-head capability gating."""

    def __init__(
        self, *, manifest: Mapping[str, object], model, context_dim: int
    ) -> None:
        self.manifest = dict(manifest)
        self._model = model
        self.model_id = str(manifest["model_id"])
        self.model_hash = str(manifest["model_hash"])
        self.feature_spec_hash = str(manifest["feature_spec_sha256"])
        self.context_dim = context_dim
        caps = manifest.get("capabilities", {})
        self._capabilities = {str(k): str(v) for k, v in caps.items()}

    @classmethod
    def from_bundle(cls, bundle_dir: str | Path) -> "ControlPredictor":
        from pes2ts_core.pesn.models.control_heads import ControlNet

        manifest, root = read_bundle_manifest(bundle_dir)
        if manifest.get("bundle_format") != SUPPORTED_BUNDLE_FORMAT_V2:
            raise BundleError(
                "BUNDLE_FORMAT_UNKNOWN",
                "control predictor requires a pesn_bundle_format_v2 bundle",
            )
        config = manifest.get("model_config")
        if not isinstance(config, dict):
            raise BundleError("MANIFEST_MALFORMED", "bundle lacks model_config")
        context_dim = int(config.get("context_dim", CONTROL_CONTEXT_DIM))
        model = ControlNet.build(
            hidden=int(config["hidden"]),
            n_layers=int(config["n_layers"]),
            n_basis=int(config["n_basis"]),
            context_dim=context_dim,
            use_attention=bool(config.get("use_coordinate_attention", False)),
        )
        states = load_bundle_weights(root)
        model.load_state_dicts(states["model_state"])
        model.train_mode(False)
        return cls(manifest=manifest, model=model, context_dim=context_dim)

    def capabilities(self) -> dict[str, str]:
        return dict(self._capabilities)

    def _head_on(self, capability: str) -> bool:
        return self._capabilities.get(capability, "disabled") != "disabled"

    def predict(
        self,
        control_input: ControlInput,
        candidate_actions: Sequence[Sequence[float]],
    ) -> ControlPrediction:
        """Residual proposal + Q_proxy for one state and its candidate list."""
        if control_input.schema_version != _CONTROL_INPUT_SCHEMA:
            raise BundleError(
                "SCHEMA_UNKNOWN",
                f"control input schema {control_input.schema_version!r} unsupported",
            )
        n_drivers = len(control_input.spec.driver_ids)
        for candidate in candidate_actions:
            if len(candidate) != n_drivers:
                raise BundleError(
                    "CANDIDATE_SHAPE",
                    "candidate vector length does not match the residual spec",
                )
        base = control_input.base_input
        features = build_features(base)
        torch = self._model.torch
        context = torch.as_tensor(
            np.asarray(control_input.context.to_vector(), dtype=np.float32
        ).reshape(1, self.context_dim)
        )
        residual_on = self._head_on(CAP_RESIDUAL)
        q_on = self._head_on(CAP_Q_PROXY)

        with torch.no_grad():
            outputs = self._model.forward_residual([features], context)
            u_flat = outputs["residual_components"].cpu().numpy().astype(float)
            counts = [int(c) for c in outputs["driver_counts"].cpu().numpy()]
        n_model_drivers = counts[0] if counts else 0

        q_estimates: list[CandidateQEstimate] = []
        if q_on and candidate_actions and n_model_drivers == n_drivers:
            with torch.no_grad():
                q_values = self._score_candidates(
                    features, context, candidate_actions, n_drivers
                )
            for value in q_values:
                finite = bool(np.isfinite(value))
                q_estimates.append(
                    CandidateQEstimate(
                        available=finite,
                        value=float(value) if finite else None,
                        reason=None if finite else "no_action_support",
                    )
                )
        else:
            q_estimates = [
                CandidateQEstimate(
                    available=False, value=None, reason="no_action_support"
                )
                for _ in candidate_actions
            ]

        residual_available = (
            residual_on
            and n_model_drivers == n_drivers
            and n_drivers > 0
            and bool(np.all(np.isfinite(u_flat)))
        )
        prediction = ControlPrediction(
            schema_version=CONTROL_PREDICTION_SCHEMA,
            input_hash=control_input.input_hash,
            model_id=self.model_id,
            model_hash=self.model_hash,
            feature_hash=self.feature_spec_hash,
            residual=tuple(float(v) for v in u_flat[:n_drivers]),
            residual_available=residual_available,
            residual_reason=None if residual_available else (
                "head_disabled" if not residual_on else "no_action_support"
            ),
            q_proxy=tuple(q_estimates),
        )
        prediction.validate()
        return prediction

    def propose_residual(
        self, control_input: ControlInput
    ) -> tuple[bool, tuple[float, ...] | None, str | None]:
        """Residual-head-only proposal: (available, u, unavailable_reason).

        Runs the residual head without scoring any candidates so the BC
        control group and the residual+Q group can both inject the frozen
        proposal into candidate generation.  Availability follows the same
        frozen gates as ``predict``: capability on, driver count match,
        finite outputs.  A non-finite or disabled proposal returns
        ``head_disabled``/``no_action_support`` — never silent zeros.
        """
        if control_input.schema_version != _CONTROL_INPUT_SCHEMA:
            raise BundleError(
                "SCHEMA_UNKNOWN",
                f"control input schema {control_input.schema_version!r} unsupported",
            )
        n_drivers = len(control_input.spec.driver_ids)
        residual_on = self._head_on(CAP_RESIDUAL)
        if not residual_on:
            return False, None, "head_disabled"
        base = control_input.base_input
        features = build_features(base)
        torch = self._model.torch
        context = torch.as_tensor(
            np.asarray(control_input.context.to_vector(), dtype=np.float32
                        ).reshape(1, self.context_dim)
        )
        with torch.no_grad():
            outputs = self._model.forward_residual([features], context)
            u_flat = outputs["residual_components"].cpu().numpy().astype(float)
            counts = [int(c) for c in outputs["driver_counts"].cpu().numpy()]
        n_model_drivers = counts[0] if counts else 0
        if n_model_drivers != n_drivers or n_drivers == 0:
            return False, None, "no_action_support"
        proposal = tuple(float(v) for v in u_flat[:n_drivers])
        if not bool(np.all(np.isfinite(proposal))):
            return False, None, "no_action_support"
        return True, proposal, None

    def _score_candidates(
        self,
        features,
        context,
        candidate_actions: Sequence[Sequence[float]],
        n_drivers: int,
    ) -> list[float]:
        torch = self._model.torch
        rows: list[float] = []
        slots: list[int] = []
        driver_rows: list[int] = []
        for slot, candidate in enumerate(candidate_actions):
            for driver_row, u_j in enumerate(candidate):
                rows.append(float(u_j))
                slots.append(slot)
                driver_rows.append(driver_row)
        q = self._model.forward_q(
            [features],
            context,
            torch.as_tensor(rows, dtype=torch.float32),
            torch.as_tensor(slots, dtype=torch.long),
            torch.as_tensor(driver_rows, dtype=torch.long),
            len(candidate_actions),
        )
        return [float(v) for v in q.cpu().numpy()]


__all__ = ["CONTROL_PREDICTION_SCHEMA", "ControlPredictor"]
