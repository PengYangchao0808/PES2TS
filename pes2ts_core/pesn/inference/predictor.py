"""Runtime predictor: qualified-input research predictions (B6).

``Predictor.from_bundle`` loads a validated whitelist bundle and returns a
predictor whose open heads follow the manifest's frozen capability table.
``predict_point`` / ``predict_batch`` share semantics and preserve order;
invalid inputs raise typed rejections (never silent row drops), unavailable
heads return masked estimates with typed reasons, and no STOP/MOVE/Q head is
ever emitted in the first-cycle research package.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Mapping, Sequence

import numpy as np

from pes2ts_core.pesn.contracts.inputs import PESNInput
from pes2ts_core.pesn.contracts.predictions import (
    CAP_COORDINATE_ACTIVITY,
    CAP_DISABLED,
    CAP_PATH_PROXIMITY,
    CAP_PROGRESS,
    CAP_TEACHER_DIRECTION,
    ScalarEstimate,
    TeacherPrediction,
    VectorEstimate,
    direction_available,
)
from pes2ts_core.pesn.bundle.reader import (
    bundle_model_config,
    load_bundle_weights,
    read_bundle_manifest,
)
from pes2ts_core.pesn.features.graph import build_features

PREDICTION_SCHEMA: Final[str] = "pesn_teacher_prediction_v1"
MIN_NORM: Final[float] = 1e-6


class Predictor:
    """Loaded research model with per-head availability from the bundle."""

    def __init__(self, *, manifest: Mapping[str, object], model, feature_spec_hash: str) -> None:
        self.manifest = dict(manifest)
        self._model = model
        self.model_id = str(manifest["model_id"])
        self.model_hash = str(manifest["model_hash"])
        self.feature_spec_hash = feature_spec_hash
        caps = manifest.get("capabilities", {})
        self._capabilities = {str(k): str(v) for k, v in caps.items()}

    @classmethod
    def from_bundle(
        cls, bundle_dir: str | Path, *, expected_feature_spec: str | None = None
    ) -> "Predictor":
        from pes2ts_core.pesn.models.encoder import SmallGraphEncoder
        from pes2ts_core.pesn.models.heads import MultiTaskTeacher, ResearchHeads

        manifest, root = read_bundle_manifest(bundle_dir)
        feature_spec_hash = str(manifest["feature_spec_sha256"])
        if expected_feature_spec is not None and expected_feature_spec != feature_spec_hash:
            from pes2ts_core.pesn.bundle.reader import BundleError

            raise BundleError(
                "FEATURE_SPEC_MISMATCH",
                "bundle feature spec does not match the caller's expectation",
            )
        config = bundle_model_config(manifest)
        encoder = SmallGraphEncoder(
            hidden=config["hidden"], n_layers=config["n_layers"], n_basis=config["n_basis"]
        )
        heads = ResearchHeads(
            feature_dim=3 * config["hidden"] + 1, hidden=config["hidden"]
        )
        model = MultiTaskTeacher(encoder, heads)
        states = load_bundle_weights(root)
        model.load_state_dicts(states["model_state"])
        model.train_mode(False)
        return cls(manifest=manifest, model=model, feature_spec_hash=feature_spec_hash)

    def capabilities(self) -> dict[str, str]:
        return dict(self._capabilities)

    def _head_on(self, capability: str) -> bool:
        return self._capabilities.get(capability, CAP_DISABLED) != CAP_DISABLED

    def predict_point(self, prediction_input: PESNInput) -> TeacherPrediction:
        return self.predict_batch([prediction_input])[0]

    def predict_batch(
        self, prediction_inputs: Sequence[PESNInput]
    ) -> list[TeacherPrediction]:
        if not prediction_inputs:
            return []
        features = [build_features(prediction_input) for prediction_input in prediction_inputs]
        torch = self._model.torch
        with torch.no_grad():
            outputs = self._model.forward(features)
        s_hat = outputs["signed_progress"].cpu().numpy().astype(float)
        score = outputs["path_score"].cpu().numpy().astype(float)
        u_flat = outputs["direction_components"].cpu().numpy().astype(float)
        act_logits = outputs["activity_logits"].cpu().numpy().astype(float)
        counts = [int(c) for c in outputs["driver_counts"].cpu().numpy()]

        predictions: list[TeacherPrediction] = []
        start = 0
        for index, prediction_input in enumerate(prediction_inputs):
            count = counts[index]
            end = start + count
            progress_on = self._head_on(CAP_PROGRESS)
            proximity_on = self._head_on(CAP_PATH_PROXIMITY)
            direction_on = self._head_on(CAP_TEACHER_DIRECTION)
            activity_on = self._head_on(CAP_COORDINATE_ACTIVITY)

            segment = u_flat[start:end]
            norm = float(np.linalg.norm(segment)) if count else 0.0
            direction_valid = direction_on and direction_available(segment.tolist())
            activity_segment = act_logits[start:end]

            prediction = TeacherPrediction(
                schema_version=PREDICTION_SCHEMA,
                input_hash=prediction_input.input_hash,
                model_id=self.model_id,
                model_hash=self.model_hash,
                feature_hash=self.feature_spec_hash,
                signed_progress=ScalarEstimate(
                    available=progress_on,
                    value=float(s_hat[index]) if progress_on else None,
                    reason=None if progress_on else "head_disabled",
                ),
                path_proximity=ScalarEstimate(
                    available=proximity_on,
                    value=abs(float(s_hat[index])) if proximity_on else None,
                    reason=None if proximity_on else "head_disabled",
                ),
                path_score=ScalarEstimate(
                    available=progress_on,
                    value=float(score[index]) if progress_on else None,
                    reason=None if progress_on else "head_disabled",
                ),
                teacher_direction=VectorEstimate(
                    available=direction_valid,
                    values=tuple(float(v) for v in (segment / norm if direction_valid else np.zeros(count))),
                    mask=tuple(direction_valid for _ in range(count)),
                    reason=None if direction_valid else (
                        "direction_norm_invalid" if direction_on and norm <= MIN_NORM else "head_disabled"
                    ),
                ),
                coordinate_activity=VectorEstimate(
                    available=activity_on and count > 0,
                    values=tuple(
                        float(1.0 / (1.0 + np.exp(-v))) for v in activity_segment
                    )
                    if activity_on and count
                    else tuple(0.0 for _ in range(count)),
                    mask=tuple(activity_on and count > 0 for _ in range(count)),
                    reason=None if activity_on and count else (
                        "coordinate_count_zero" if activity_on else "head_disabled"
                    ),
                ),
            )
            prediction.validate()
            predictions.append(prediction)
            start = end
        return predictions


__all__ = ["MIN_NORM", "PREDICTION_SCHEMA", "Predictor"]
